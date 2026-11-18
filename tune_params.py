"""Сетка параметров на 20 товарах dev. По одному параметру от базовой точки.

После сетки выбранная комбинация проверяется на всех 40 товарах dev.
Если доля валидных ниже 80%, число попыток поднимается до 3 и проверка повторяется.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from pathlib import Path

from cardgen.generate import CardGenerator, GenParams
from cardgen.io_utils import load_jsonl, write_json
from cardgen.metrics import evaluate_predictions
from cardgen.runner import ROOT, execute, load_tokenizer, make_client, take_subset
from cardgen.generate import public_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подбор параметров генерации")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-probe", action="store_true")
    return parser.parse_args()


def grid(base: GenParams) -> list[tuple[str, object, dict]]:
    """Параметр, значение, отличия от базы. Пустой словарь — сама базовая точка."""
    return [
        ("temperature", 0.0, {"temperature": 0.0}),
        ("temperature", 0.3, {}),
        ("temperature", 0.7, {"temperature": 0.7}),
        ("top_p", 0.9, {}),
        ("top_p", 1.0, {"top_p": 1.0}),
        ("top_k", 20, {"top_k": 20}),
        ("top_k", 50, {}),
        ("top_k", "unlimited", {"top_k": None}),
        ("repetition_penalty", 1.0, {"repetition_penalty": 1.0}),
        ("repetition_penalty", 1.1, {}),
        ("repetition_penalty", 1.2, {"repetition_penalty": 1.2}),
        ("max_tokens", 250, {"max_tokens": 250}),
        ("max_tokens", 350, {}),
        ("max_tokens", 500, {"max_tokens": 500}),
        ("json_mode", "guided", {}),
        ("json_mode", "object", {"json_mode": "object"}),
        ("json_mode", "off", {"json_mode": "off"}),
    ]


def summarize(products: list[dict], results: list[dict]) -> dict:
    quality = evaluate_predictions(products, [public_card(result) for result in results])
    n = len(results)
    truncated = sum(1 for result in results if "length" in (result.get("finish_reasons") or []))
    return {
        "valid_rate": round(quality["valid_rate"], 4),
        "avg_description_len": round(quality["avg_description_len"], 1),
        "avg_output_tokens": round(sum(result["output_tokens"] for result in results) / n, 1),
        "avg_attempts": round(sum(result["attempts"] for result in results) / n, 3),
        "truncation_rate": round(truncated / n, 4),
        "avg_latency": round(sum(result["latency_sec"] for result in results) / n, 3),
        "errors_by_reason": quality["errors_by_reason"],
        "n": n,
        "valid": quality["valid"],
    }


def choose_value(parameter: str, rows: list[dict]) -> dict:
    if parameter == "max_tokens":
        intact = [row for row in rows if row["truncation_rate"] == 0] or rows
        best_valid = max(row["valid_rate"] for row in intact)
        near = [row for row in intact if row["valid_rate"] >= best_valid - 0.02]
        return min(near, key=lambda row: (row["value"], row["avg_output_tokens"]))
    best_valid = max(row["valid_rate"] for row in rows)
    near = [row for row in rows if row["valid_rate"] >= best_valid - 0.02]
    if parameter == "json_mode":
        return min(near, key=lambda row: (row["avg_latency"], row["avg_output_tokens"]))
    return min(near, key=lambda row: (row["avg_output_tokens"], row["avg_latency"]))


def apply_choice(base: GenParams, chosen: dict[str, dict]) -> GenParams:
    updates = {}
    for parameter, row in chosen.items():
        value = None if row["value"] == "unlimited" else row["value"]
        updates[parameter] = value
    return replace(base, **updates)


async def run_config(args, client, tokenizer, products, params: GenParams, probe: bool) -> dict:
    generator = CardGenerator(client, tokenizer, args.model, params)
    if probe:
        await generator.probe_json_mode()
    results, elapsed = await execute(products, generator, args.concurrency, progress=False)
    summary = summarize(products, results)
    summary["elapsed_sec"] = round(elapsed, 3)
    summary["json_mode_effective"] = generator.params.json_mode
    return summary


async def confirm(args, client, tokenizer, products, params: GenParams) -> dict:
    print(
        f"проверка на {len(products)} товарах dev, попыток до {params.max_attempts}, "
        f"temperature {params.temperature}, max_tokens {params.max_tokens}, json {params.json_mode}",
        flush=True,
    )
    summary = await run_config(args, client, tokenizer, products, params, probe=False)
    print(
        f"dev: валидных {summary['valid_rate']:.1%}, "
        f"длина описания {summary['avg_description_len']:.0f}, "
        f"выход {summary['avg_output_tokens']:.0f} токенов, брак {summary['errors_by_reason']}",
        flush=True,
    )
    return summary


async def main_async(args: argparse.Namespace) -> None:
    rows_all = load_jsonl(args.input)
    sample = take_subset(rows_all, args.n)
    client = make_client(args.base_url, args.api_key, timeout=180)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    base = GenParams()
    cache: dict[tuple, dict] = {}
    table = []
    probed = False
    for parameter, value, overrides in grid(base):
        signature = tuple(sorted((key, repr(item)) for key, item in overrides.items()))
        if signature not in cache:
            params = replace(base, **overrides)
            print(f"эксперимент {parameter}={value}", flush=True)
            try:
                cache[signature] = await run_config(
                    args, client, tokenizer, sample, params, probe=not probed
                )
            except Exception as exc:
                print(f"  эксперимент не выполнился: {exc}", flush=True)
                cache[signature] = {
                    "valid_rate": 0.0,
                    "avg_description_len": 0.0,
                    "avg_output_tokens": 0.0,
                    "avg_attempts": 0.0,
                    "truncation_rate": 1.0,
                    "avg_latency": 0.0,
                    "errors_by_reason": {"эксперимент": str(exc)},
                    "n": len(sample),
                    "valid": 0,
                    "elapsed_sec": 0.0,
                    "json_mode_effective": params.json_mode,
                }
            probed = True
        row = dict(cache[signature])
        row["parameter"] = parameter
        row["value"] = value
        table.append(row)
        print(
            f"  валидных {row['valid_rate']:.1%}, описание {row['avg_description_len']:.0f} симв., "
            f"выход {row['avg_output_tokens']:.0f} токенов, обрывы {row['truncation_rate']:.0%}",
            flush=True,
        )

    grouped: dict[str, list[dict]] = {}
    for row in table:
        grouped.setdefault(row["parameter"], []).append(row)
    chosen_rows = {parameter: choose_value(parameter, items) for parameter, items in grouped.items()}
    chosen = apply_choice(base, chosen_rows)
    print("выбрано:", {key: getattr(chosen, key) for key in (
        "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "json_mode"
    )}, flush=True)

    confirmation = await confirm(args, client, tokenizer, rows_all, chosen)
    if confirmation["valid_rate"] < 0.8 and chosen.max_attempts < 3:
        chosen = replace(chosen, max_attempts=3)
        print("доля ниже 80%, повторяю проверку с 3 попытками", flush=True)
        confirmation = await confirm(args, client, tokenizer, rows_all, chosen)
    if confirmation["valid_rate"] < 0.8 and chosen.temperature != 0.0:
        chosen = replace(chosen, temperature=0.0, max_attempts=max(chosen.max_attempts, 3))
        print("доля всё ещё ниже 80%, пробую temperature 0", flush=True)
        confirmation = await confirm(args, client, tokenizer, rows_all, chosen)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "tune.json",
        {
            "rows": table,
            "chosen": {key: getattr(chosen, key) for key in (
                "temperature", "top_p", "top_k", "repetition_penalty",
                "max_tokens", "max_attempts", "json_mode",
            )},
            "confirmation": confirmation,
        },
    )
    payload = asdict(chosen)
    write_json(args.output_dir / "best_params.json", payload)
    if confirmation["valid_rate"] < 0.8:
        raise SystemExit(
            f"на dev валидных {confirmation['valid_rate']:.1%} — ниже 80%. "
            "Смотрите outputs/tune.json и поправьте промпт."
        )


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
