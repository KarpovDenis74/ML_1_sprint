"""Пакетная генерация карточек через локальный vLLM.

Примеры:
  python run_batch.py --input data/benchmark.jsonl --concurrency 16
  python run_batch.py --sweep 1,4,16,32 --sweep-n 40
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cardgen.generate import CardGenerator
from cardgen.io_utils import load_jsonl, write_json
from cardgen.runner import (
    ROOT,
    execute,
    load_params,
    load_tokenizer,
    long_input_stats,
    make_client,
    split_results,
    take_subset,
    write_run_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пакетная генерация карточек на vLLM")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "benchmark.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer", default=None, help="По умолчанию совпадает с --model")
    parser.add_argument("--params", type=Path, default=ROOT / "outputs" / "best_params.json")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", default=None, help="Число или unlimited")
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--json-mode", choices=("guided", "object", "off"), default=None)
    parser.add_argument("--sweep", default="", help="Список concurrency через запятую, например 1,4,16,32")
    parser.add_argument("--sweep-n", type=int, default=40)
    parser.add_argument("--no-probe", action="store_true")
    return parser.parse_args()


def apply_cli(params, args):
    if args.temperature is not None:
        params.temperature = args.temperature
    if args.top_p is not None:
        params.top_p = args.top_p
    if args.top_k is not None:
        params.top_k = None if args.top_k == "unlimited" else int(args.top_k)
    if args.repetition_penalty is not None:
        params.repetition_penalty = args.repetition_penalty
    if args.max_tokens is not None:
        params.max_tokens = args.max_tokens
    if args.max_attempts is not None:
        params.max_attempts = args.max_attempts
    if args.json_mode is not None:
        params.json_mode = args.json_mode
    return params


def choose_concurrency(rows: list[dict]) -> int:
    """Берёт точку, где пропускная способность уже выросла, а доля валидных не просела."""
    if not rows:
        return 8
    ordered = sorted(rows, key=lambda row: row["concurrency"])
    baseline = ordered[0]["valid_rate"]
    usable = [row for row in ordered if row["valid_rate"] >= baseline - 0.08] or ordered
    chosen = usable[0]
    for row in usable[1:]:
        if chosen["throughput"] <= 0:
            chosen = row
            continue
        gain = (row["throughput"] - chosen["throughput"]) / chosen["throughput"]
        if gain >= 0.08:
            chosen = row
    return int(chosen["concurrency"])


async def run_sweep(args, params, products: list[dict]) -> None:
    subset = take_subset(products, args.sweep_n)
    levels = [int(item) for item in args.sweep.split(",") if item.strip()]
    client = make_client(args.base_url, args.api_key, timeout=180)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    rows = []
    print(f"sweep: {len(subset)} товаров, concurrency {levels}", flush=True)
    for level in levels:
        generator = CardGenerator(client, tokenizer, args.model, params)
        if not args.no_probe and level == levels[0]:
            await generator.probe_json_mode()
        results, elapsed = await execute(subset, generator, level)
        predictions, details, _ = split_results(results)
        from cardgen.metrics import evaluate_predictions

        quality = evaluate_predictions(subset, predictions)
        throughput = len(subset) / elapsed if elapsed else 0.0
        latency = sum(item["latency_sec"] for item in details) / len(details)
        row = {
            "concurrency": level,
            "n": len(subset),
            "elapsed_sec": round(elapsed, 3),
            "throughput": round(throughput, 4),
            "avg_latency": round(latency, 3),
            "valid_rate": round(quality["valid_rate"], 4),
            "valid": quality["valid"],
            "time_for_300_sec": round(300 / throughput, 1) if throughput else None,
        }
        rows.append(row)
        print(
            f"concurrency {level}: throughput {row['throughput']:.3f} карт/с, "
            f"latency {row['avg_latency']:.2f} с, валидных {row['valid_rate']:.1%}, "
            f"оценка на 300 — {row['time_for_300_sec']} с",
            flush=True,
        )
    chosen = choose_concurrency(rows)
    payload = {"rows": rows, "chosen_concurrency": chosen, "n": len(subset)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "sweep.json", payload)
    print(f"выбрана concurrency {chosen}", flush=True)


async def run_full(args, params, products: list[dict]) -> None:
    concurrency = args.concurrency
    sweep_path = args.output_dir / "sweep.json"
    if concurrency is None and sweep_path.exists():
        concurrency = int(load_json_concurrency(sweep_path))
    if concurrency is None:
        concurrency = 16
    client = make_client(args.base_url, args.api_key, timeout=180)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    generator = CardGenerator(client, tokenizer, args.model, params)
    if not args.no_probe:
        await generator.probe_json_mode()
    print(
        f"прогон {len(products)} товаров, concurrency {concurrency}, "
        f"попыток до {params.max_attempts}, json {params.json_mode}",
        flush=True,
    )
    results, elapsed = await execute(products, generator, concurrency)
    write_run_files(
        args.output_dir,
        products=products,
        results=results,
        elapsed_sec=elapsed,
        model=args.model,
        concurrency=concurrency,
        params=params,
        json_mode=generator.params.json_mode,
    )
    valid = sum(1 for result in results if result["valid"])
    stats = long_input_stats(split_results(results)[1])
    print(
        f"готово за {elapsed:.1f} с, валидных {valid}/{len(products)} "
        f"({valid / len(products):.1%}). Максимум токенов входа {stats.get('max_first_prompt_tokens')}.",
        flush=True,
    )


def load_json_concurrency(path: Path) -> int:
    import json

    return int(json.loads(path.read_text(encoding="utf-8"))["chosen_concurrency"])


def main() -> None:
    args = parse_args()
    params = apply_cli(load_params(args.params), args)
    products = load_jsonl(args.input)
    if args.limit:
        products = take_subset(products, args.limit)
    if args.sweep:
        asyncio.run(run_sweep(args, params, products))
    else:
        asyncio.run(run_full(args, params, products))


if __name__ == "__main__":
    main()
