"""Та же генерация карточек через OpenRouter. Цена считается по токенам ответа API.

Ключ только из окружения:
  export OPENROUTER_API_KEY=...
  python compare_api.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dataclasses import replace

from cardgen.generate import CardGenerator
from cardgen.io_utils import load_jsonl, write_json, write_jsonl
from cardgen.metrics import api_comparison_from_run
from cardgen.runner import ROOT, execute, load_params, load_tokenizer, make_client, split_results, take_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнение с моделью через OpenRouter")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b"),
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--params", type=Path, default=ROOT / "outputs" / "best_params.json")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--no-probe", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "Нет OPENROUTER_API_KEY. Задайте ключ в окружении, в код его писать нельзя."
        )
    products = take_subset(load_jsonl(args.input), args.n)
    params = load_params(args.params)
    # Промпт тот же. Параметры vLLM уйдут в запрос и будут сняты, если API ответит 400.
    params = replace(params, json_mode="object")
    client = make_client(
        args.base_url,
        api_key,
        timeout=120,
        headers={
            "HTTP-Referer": "https://praktikum.yandex.ru",
            "X-Title": "polka-cardgen",
        },
    )
    tokenizer = load_tokenizer(args.tokenizer)
    generator = CardGenerator(
        client,
        tokenizer,
        args.model,
        params,
        handle_http_errors=True,
    )
    if not args.no_probe:
        await generator.probe_json_mode()
    print(
        f"API {args.model}: {len(products)} товаров, concurrency {args.concurrency}",
        flush=True,
    )
    results, elapsed = await execute(products, generator, args.concurrency)
    predictions, details, invalid = split_results(results)
    comparison = api_comparison_from_run(args.model, products, predictions, details, elapsed)
    comparison["http_errors"] = generator.http_errors
    comparison["json_mode"] = generator.params.json_mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "api_predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "api_details.jsonl", details)
    write_json(args.output_dir / "api_invalid_samples.json", {"items": invalid[:20]})
    write_json(args.output_dir / "api_comparison.json", comparison)
    print(
        f"API валидных {comparison['valid_rate']:.1%}, "
        f"вход {comparison['input_tokens']} токенов, выход {comparison['output_tokens']}, "
        f"{comparison['cost_per_1000']:.2f} ₽/1000, ошибки HTTP {generator.http_errors}",
        flush=True,
    )
    return comparison


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
