#!/usr/bin/env python3
# Detect corrupted model outputs and evaluate ReRun decisions.
# should_aborted and is_aborted are ignored. Actual rerun is inferred from
# whether a sample_id has more than one record.

import argparse
import hashlib
import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_API_KEY = "sk-4150da41c4b043da8a509174c8af2309"


def is_garbled(text: str) -> bool:
    if not text:
        return True

    words = re.findall(r"\w+", text)
    max_repeat = 1
    cur = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            cur += 1
            max_repeat = max(max_repeat, cur)
        else:
            cur = 1
    if max_repeat > 15:
        return True

    if re.search(r"(.)\1{30,}", text):
        return True

    if re.search(r"(.{2,9})\1{8,}", text):
        return True

    lines_list = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines_list) > 8:
        unique = len(set(lines_list))
        if unique < len(lines_list) * 0.4:
            return True

    cjk = len(re.findall(r"[一-鿿]", text))
    alpha = len(re.findall(r"[a-zA-Z]", text))
    if (cjk > 10 and alpha > 50) or (cjk >= 2 and alpha > 20 and len(text) > 50):
        return True

    return False


def get_generated_text(sample: dict) -> str:
    thinking = sample.get("thinking_process", "") or ""
    answer = sample.get("generated_answer", "") or ""
    return (thinking + "\n" + answer).strip()


def threshold_from_name(name: str) -> str:
    match = re.search(r"sr<=([0-9.]+)_cons", name)
    return match.group(1) if match else "unknown"


def oracle_id(file_path: str, sample_id: str) -> str:
    return f"{threshold_from_name(Path(file_path).name)}::{sample_id}"


def row_oracle_id(row: dict) -> str:
    return f"{row['threshold']}::{row['sample_id']}"


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def check_sample_rule(sample: dict) -> dict:
    text = get_generated_text(sample)
    should_abort = is_garbled(text)
    return {
        "sample_id": str(sample.get("sample_id", "")),
        "token_length": sample.get("token_length", 0),
        "should_rerun": should_abort,
        "output_garbled": should_abort,
        "reason": "rule_garbled" if should_abort else "rule_ok",
        "confidence": 1.0,
        "detector": "rule",
        "cached": False,
    }


class LLMDetector:
    def __init__(self, cache_path=None, model="deepseek-v4-flash", base_url="https://api.deepseek.com", max_chars=12000):
        self.cache_path = cache_path
        self.model = model
        self.base_url = base_url
        self.max_chars = max_chars
        self.cache = {}
        self._cache_lock = threading.Lock()
        if cache_path:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            self.cache[item["fingerprint"]] = item["result"]
            except FileNotFoundError:
                pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI SDK is required for LLM detection.") from exc

        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY") or DEFAULT_API_KEY
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def detect(self, sample: dict) -> dict:
        text = get_generated_text(sample)
        fingerprint = text_fingerprint(text)
        cached = self.cache.get(fingerprint)
        if cached is not None:
            result = dict(cached)
            result["cached"] = True
            return result

        result = self._call_single(text)
        result["detector"] = "llm"
        result["cached"] = False
        self._save_cache(fingerprint, result)
        return result

    def detect_batch(self, items: list[dict]) -> list[dict]:
        if not items:
            return []
        prompt_items = []
        for item in items:
            text = item["text"][: self.max_chars]
            if len(item["text"]) > self.max_chars:
                text += "\n\n[TRUNCATED FOR DETECTION]"
            prompt_items.append(
                {
                    "id": item["oracle_id"],
                    "threshold": item.get("threshold", "unknown"),
                    "sample_id": item["sample_id"],
                    "token_length": item.get("token_length", 0),
                    "generated_output": text,
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You judge whether model generations are corrupted enough that inference should be aborted and rerun.",
                },
                {
                    "role": "user",
                    "content": (
                        "Judge only generation corruption, not math correctness.\n"
                        "Abort only when reasoning has truly collapsed: severe endless repetition, unrecoverable garbled tokens, nonsensical degeneration, or obvious looping.\n"
                        "Do not abort for merely verbose, mathematically wrong, coherent but incomplete reasoning, or mixed/garbled characters if the reasoning still keeps progressing and can plausibly produce an answer.\n\n"
                        "Return JSON only with this schema:\n"
                        "{\"labels\": [{\"id\": \"threshold::sample_id\", \"should_abort\": true/false, \"reason\": \"...\", \"confidence\": 0.0}]}\n\n"
                        "Items:\n"
                        + json.dumps(prompt_items, ensure_ascii=False)
                    ),
                },
            ],
            stream=False,
            response_format={"type": "json_object"},
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        content = response.choices[0].message.content or ""
        parsed = self._parse_json(content)
        labels = parsed.get("labels", []) if isinstance(parsed, dict) else parsed
        by_id = {str(label.get("id")): label for label in labels}

        results = []
        for item in items:
            label = by_id.get(item["oracle_id"])
            if label is None:
                try:
                    fallback = self._call_single(item["text"])
                    fallback["detector"] = "llm_single_fallback"
                    fallback["cached"] = False
                    results.append(fallback)
                except Exception as exc:
                    print(f"WARNING: fallback single-call failed for {item['oracle_id']}: {exc}", file=sys.stderr)
                    results.append({
                        "should_rerun": False,
                        "reason": f"fallback_error: {exc}",
                        "confidence": 0.0,
                        "detector": "llm_single_fallback_error",
                        "cached": False,
                    })
                continue
            results.append(
                {
                    "should_rerun": bool(label.get("should_abort", label.get("should_rerun", False))),
                    "reason": str(label.get("reason", "")),
                    "confidence": float(label.get("confidence", 0.0)),
                    "detector": "llm_batch",
                    "cached": False,
                }
            )
        return results

    def _call_single(self, text: str) -> dict:
        clipped = text[: self.max_chars]
        if len(text) > self.max_chars:
            clipped += "\n\n[TRUNCATED FOR DETECTION]"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You judge whether a model generation is corrupted enough that inference should be aborted and rerun.",
                },
                {
                    "role": "user",
                    "content": (
                        "Judge only generation corruption, not math correctness.\n"
                        "Abort only when reasoning has truly collapsed: severe endless repetition, unrecoverable garbled tokens, nonsensical degeneration, or obvious looping.\n"
                        "Do not abort for merely verbose, mathematically wrong, coherent but incomplete reasoning, or mixed/garbled characters if the reasoning still keeps progressing and can plausibly produce an answer.\n\n"
                        "Return JSON only with this schema:\n"
                        "{\"should_abort\": true/false, \"reason\": \"...\", \"confidence\": 0.0}\n\n"
                        f"Generated output:\n{clipped}"
                    ),
                },
            ],
            stream=False,
            response_format={"type": "json_object"},
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        data = self._parse_json(response.choices[0].message.content or "")
        return {
            "should_rerun": bool(data.get("should_abort", data.get("should_rerun", False))),
            "reason": str(data.get("reason", "")),
            "confidence": float(data.get("confidence", 0.0)),
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first balanced JSON object from text, handling nesting and strings."""
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return ""

    @staticmethod
    def _parse_json(content: str):
        text = content.strip()
        # Remove markdown code fences (handles trailing text after closing ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```.*$", "", text, flags=re.S)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        extracted = LLMDetector._extract_json(text)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        print(f"ERROR: LLM response is not JSON. Content (first 500): {content[:500]}", file=sys.stderr)
        raise ValueError(f"LLM response is not JSON: {content[:500]}")

    def _save_cache(self, fingerprint: str, result: dict) -> None:
        if not self.cache_path:
            return
        with self._cache_lock:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"fingerprint": fingerprint, "result": result}, ensure_ascii=False) + "\n")
            self.cache[fingerprint] = result


def load_records(filepath: str) -> dict[str, list[dict]]:
    records = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                records[str(obj["sample_id"])].append(obj)
    return records


def check_sample(sample: dict, detector=None) -> dict:
    if detector is None:
        return check_sample_rule(sample)
    result = detector.detect(sample)
    return {
        "sample_id": str(sample.get("sample_id", "")),
        "token_length": sample.get("token_length", 0),
        "should_rerun": result["should_rerun"],
        "output_garbled": result["should_rerun"],
        "reason": result.get("reason", ""),
        "confidence": result.get("confidence", 0.0),
        "detector": result.get("detector", "llm"),
        "cached": result.get("cached", False),
    }


def evaluate(filepath: str, detector=None) -> dict:
    records = load_records(filepath)
    oracle = {}
    for sid, recs in records.items():
        result = check_sample(recs[0], detector=detector)
        oracle[oracle_id(filepath, sid)] = result["should_rerun"]
    return evaluate_with_oracle(filepath, oracle)


def evaluate_with_oracle(filepath: str, oracle: dict[str, bool]) -> dict:
    records = load_records(filepath)
    tp = fn = fp = tn = 0
    missed = []
    false_alarms = []
    missing_oracle = []
    for sid in sorted(records.keys(), key=int):
        oid = oracle_id(filepath, sid)
        if oid not in oracle:
            missing_oracle.append(oid)
            continue
        recs = records[sid]
        first = recs[0]
        actually_rerun = len(recs) > 1
        should_rerun = bool(oracle[oid])
        result = {"reason": "oracle", "confidence": 1.0}
        if should_rerun and actually_rerun:
            tp += 1
        elif should_rerun and not actually_rerun:
            fn += 1
            missed.append(sample_report_item(sid, first, result))
        elif not should_rerun and actually_rerun:
            fp += 1
            false_alarms.append(sample_report_item(sid, first, result))
        else:
            tn += 1
    metrics = make_metrics(tp, fn, fp, tn)
    metrics["missed"] = missed
    metrics["false_alarms"] = false_alarms
    metrics["missing_oracle"] = missing_oracle
    return metrics


def make_metrics(tp: int, fn: int, fp: int, tn: int) -> dict:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        "total_samples": tp + fn + fp + tn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "fdr": fdr,
        "precision": precision,
        "fpr": fpr,
    }


def sample_report_item(sid: str, sample: dict, result: dict) -> dict:
    return {
        "sample_id": sid,
        "token_length": sample.get("token_length", 0),
        "reason": result.get("reason", ""),
        "confidence": result.get("confidence", 0.0),
        "thinking_process": sample.get("thinking_process", ""),
        "generated_answer": sample.get("generated_answer", ""),
        "reference_answer": sample.get("reference_answer", ""),
    }


def print_summary(metrics: dict) -> None:
    tp, fn, fp, tn = metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]
    print("=" * 60)
    print("  ReRun Mechanism Evaluation")
    print("=" * 60)
    print("                    Got ReRun  No ReRun")
    print(f"  Should ReRun       {tp:>6}     {fn:>6}    (total: {tp + fn})")
    print(f"  Should NOT ReRun   {fp:>6}     {tn:>6}    (total: {fp + tn})")
    print()
    print(f"  Recall (召回率)          : {tp}/{tp + fn} = {metrics['recall'] * 100:.2f}%")
    print(f"  False Discovery (误分率) : {fp}/{tp + fp} = {metrics['fdr'] * 100:.2f}%")
    print(f"  Precision                : {tp}/{tp + fp} = {metrics['precision'] * 100:.2f}%")
    print(f"  False Positive Rate      : {fp}/{fp + tn} = {metrics['fpr'] * 100:.2f}%")
    print()
    print(f"  Missed corrupted outputs : {fn}")
    print(f"  False alarms             : {fp}")
    if metrics.get("missing_oracle"):
        print(f"  Missing oracle labels    : {len(metrics['missing_oracle'])}")


def build_detector(args):
    if getattr(args, "detector", "rule") == "rule":
        return None
    return LLMDetector(cache_path=args.cache, model=args.model, base_url=args.base_url, max_chars=args.max_chars)


def load_oracle(path: str) -> dict[str, bool]:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                labels[row_oracle_id(obj)] = bool(obj["should_abort"])
    return labels


def _load_oracle_rows(path: str) -> list[dict]:
    """Load all rows from an oracle JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_oracle_ids(path: str) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ids.add(row_oracle_id(obj))
    return ids


def cmd_build_oracle(args) -> None:
    detector = LLMDetector(model=args.model, base_url=args.base_url, max_chars=args.max_chars)
    files = [str(Path(p)) for p in args.inputs]
    all_records = {path: load_records(path) for path in files}
    done = existing_oracle_ids(args.oracle)
    all_sids = sorted({sid for records in all_records.values() for sid in records.keys()}, key=int)
    if args.sample_start is not None:
        all_sids = [sid for sid in all_sids if int(sid) >= args.sample_start]
    if args.sample_end is not None:
        all_sids = [sid for sid in all_sids if int(sid) <= args.sample_end]
    if args.limit is not None:
        all_sids = all_sids[: args.limit]

    # Build pending work items: list of (sample_id, items_list)
    pending = []
    for sid in all_sids:
        items = []
        for path in files:
            records = all_records[path]
            if sid not in records:
                continue
            oid = oracle_id(path, sid)
            if oid in done:
                continue
            first = records[sid][0]
            items.append({
                "oracle_id": oid,
                "file": Path(path).name,
                "threshold": threshold_from_name(Path(path).name),
                "sample_id": sid,
                "token_length": first.get("token_length", 0),
                "text": get_generated_text(first),
                "actually_rerun": len(records[sid]) > 1,
            })
        if items:
            pending.append((sid, items))

    if not pending:
        print(f"All oracle_ids already in {args.oracle}, nothing to do.")
        return

    Path(args.oracle).parent.mkdir(parents=True, exist_ok=True)
    concurrency = getattr(args, "concurrency", 1)

    def process_sample(sid, items):
        """Process one sample_id batch and return (sample_id, rows) or raise."""
        results = detector.detect_batch(items)
        rows = []
        for item, result in zip(items, results):
            rows.append({
                "oracle_id": item["oracle_id"],
                "threshold": item["threshold"],
                "sample_id": item["sample_id"],
                "actually_rerun": item["actually_rerun"],
                "token_length": item["token_length"],
                "should_abort": bool(result["should_rerun"]),
                "reason": result.get("reason", ""),
                "confidence": result.get("confidence", 0.0),
                "detector": result.get("detector", "llm_batch"),
            })
        return (sid, rows)

    all_rows = []
    written = 0
    if concurrency <= 1:
        for sid, items in pending:
            _, rows = process_sample(sid, items)
            all_rows.extend(rows)
            written += len(rows)
            print(f"labeled sample_id={sid} items={len(items)} total_written={written}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_sample, sid, items): sid for sid, items in pending}
            for future in as_completed(futures):
                sid, rows = future.result()
                all_rows.extend(rows)
                written += len(rows)
                print(f"labeled sample_id={sid} items={len(rows)} total_written={written}")

    # Merge with previously existing oracle entries (preserve old + new)
    existing_rows = _load_oracle_rows(args.oracle) if Path(args.oracle).exists() else []
    existing_by_oid = {r["oracle_id"]: r for r in existing_rows}
    for row in all_rows:
        existing_by_oid[row["oracle_id"]] = row
    merged = sorted(existing_by_oid.values(), key=lambda r: (int(r["sample_id"]), float(r["threshold"])))

    with open(args.oracle, "w", encoding="utf-8") as out:
        for row in merged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"oracle written: {args.oracle}; total labels={len(merged)} (new={len(all_rows)})")


def cmd_eval_with_oracle(args) -> None:
    oracle = load_oracle(args.oracle)
    for path in args.inputs:
        metrics = evaluate_with_oracle(path, oracle)
        print(f"\nFILE {Path(path).name}")
        print_summary(metrics)


def cmd_eval(args) -> None:
    detector = build_detector(args)
    metrics = evaluate(args.input, detector=detector)
    print_summary(metrics)
    if args.output and metrics["missed"]:
        with open(args.output, "w", encoding="utf-8") as f:
            for item in metrics["missed"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n  Missed samples written to: {args.output}")
    if args.false_output and metrics["false_alarms"]:
        with open(args.false_output, "w", encoding="utf-8") as f:
            for item in metrics["false_alarms"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n  False-alarm samples written to: {args.false_output}")


def add_llm_args(parser):
    parser.add_argument("--model", default="deepseek-v4-flash", help="LLM model name")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI-compatible base URL")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars per generated output sent to LLM")


def main():
    commands = {"eval", "build-oracle", "eval-with-oracle"}
    if len(sys.argv) > 1 and sys.argv[1] not in commands and sys.argv[1] not in {"-h", "--help"}:
        sys.argv.insert(1, "eval")
    parser = argparse.ArgumentParser(description="Detect corrupted model outputs and evaluate ReRun accuracy.")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("eval", help="Evaluate one result file with rule or per-sample LLM detector")
    eval_p.add_argument("input", help="Path to JSONL file")
    eval_p.add_argument("--output", "-o", default=None, help="Output path for missed samples JSONL")
    eval_p.add_argument("--false-output", default=None, help="Output path for false-alarm samples JSONL")
    eval_p.add_argument("--detector", choices=["rule", "llm"], default="rule", help="Detector backend")
    eval_p.add_argument("--cache", default=None, help="JSONL cache for per-sample LLM decisions")
    add_llm_args(eval_p)
    eval_p.set_defaults(func=cmd_eval)

    build_p = sub.add_parser("build-oracle", help="Build LLM oracle; one API batch per sample_id across files")
    build_p.add_argument("inputs", nargs="+", help="Result JSONL files")
    build_p.add_argument("--oracle", required=True, help="Output oracle JSONL path")
    build_p.add_argument("--sample-start", type=int, default=None)
    build_p.add_argument("--sample-end", type=int, default=None)
    build_p.add_argument("--limit", type=int, default=None, help="Limit number of sample_ids for testing")
    build_p.add_argument("--concurrency", "-c", type=int, default=1, help="Number of concurrent batch API calls")
    add_llm_args(build_p)
    build_p.set_defaults(func=cmd_build_oracle)

    oracle_p = sub.add_parser("eval-with-oracle", help="Evaluate result files using a fixed oracle JSONL")
    oracle_p.add_argument("inputs", nargs="+", help="Result JSONL files")
    oracle_p.add_argument("--oracle", required=True, help="Oracle JSONL path")
    oracle_p.set_defaults(func=cmd_eval_with_oracle)

    # Backward-compatible default: detect_garbled.py <input> [options]
    parser.add_argument("legacy_input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--output", "-o", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--false-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detector", choices=["rule", "llm"], default="rule", help=argparse.SUPPRESS)
    parser.add_argument("--cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="deepseek-v4-flash", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", default="https://api.deepseek.com", help=argparse.SUPPRESS)
    parser.add_argument("--max-chars", type=int, default=12000, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.command is None:
        if not args.legacy_input:
            parser.print_help()
            return
        args.input = args.legacy_input
        cmd_eval(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
