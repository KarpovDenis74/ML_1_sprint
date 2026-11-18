"""Общие куски запуска: клиент, токенайзер, пакет, запись файлов прогона."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cardgen.batch import run_concurrent
from cardgen.generate import CardGenerator, GenParams, public_card
from cardgen.io_utils import load_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)


def make_client(base_url: str, api_key: str, timeout: float, headers: dict[str, str] | None = None) -> Any:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key, "timeout": timeout}
    if headers:
        kwargs["default_headers"] = headers
    return AsyncOpenAI(**kwargs)


async def execute(
    products: list[dict[str, Any]],
    generator: CardGenerator,
    concurrency: int,
    *,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], float]:
    done = 0
    total = len(products)
    lock = asyncio.Lock()

    async def one(product: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        result = await generator.generate_one(product)
        if progress:
            async with lock:
                done += 1
                if done == total or done % 10 == 0:
                    state = "ok" if result["valid"] else ",".join(result["errors"]) or "fail"
                    print(
                        f"{done}/{total} {result['product_id']} {state} "
                        f"{result['latency_sec']:.1f}s попыток {result['attempts']}",
                        flush=True,
                    )
        return result

    return await run_concurrent(products, concurrency, one)


def split_results(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = [public_card(result) for result in results]
    details = []
    invalid = []
    for result in results:
        raw = result.get("last_raw") or ""
        detail = {key: value for key, value in result.items() if key not in {"card", "last_raw"}}
        details.append(detail)
        if not result["valid"]:
            invalid.append(
                {
                    "product_id": result["product_id"],
                    "errors": result["errors"],
                    "error_messages": result["error_messages"],
                    "raw": raw[:2000],
                    "card": result.get("card"),
                }
            )
    return predictions, details, invalid


def long_input_stats(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        return {}
    long_rows = [row for row in details if int(row.get("reviews_total") or 0) >= 90]
    used = [int(row["reviews_used"]) for row in long_rows] or [0]
    tokens = [int(row.get("first_prompt_tokens") or 0) for row in details]
    return {
        "max_first_prompt_tokens": max(tokens),
        "n_over_1500": sum(1 for value in tokens if value > 1500),
        "n_truncated": sum(1 for row in details if row.get("truncated_input")),
        "n_long": len(long_rows),
        "avg_reviews_used_long": round(sum(used) / len(used), 1) if long_rows else 0,
        "avg_unique_long": round(
            sum(int(row.get("reviews_unique") or 0) for row in long_rows) / len(long_rows), 1
        )
        if long_rows
        else 0,
    }


def write_run_files(
    directory: Path,
    *,
    products: list[dict[str, Any]],
    results: list[dict[str, Any]],
    elapsed_sec: float,
    model: str,
    concurrency: int,
    params: GenParams,
    json_mode: str,
) -> None:
    predictions, details, invalid = split_results(results)
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "predictions.jsonl", predictions)
    write_jsonl(directory / "details.jsonl", details)
    write_json(directory / "invalid_samples.json", {"items": invalid[:30]})
    write_json(
        directory / "run_meta.json",
        {
            "model": model,
            "concurrency": concurrency,
            "params": params.as_report_dict(),
            "elapsed_sec": elapsed_sec,
            "n": len(products),
            "max_attempts": params.max_attempts,
            "json_mode": json_mode,
            "seed": params.seed,
        },
    )
    write_json(directory / "diagnostics.json", long_input_stats(details))


def take_subset(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Короткая выборка, в которой есть и обычные товары, и товары с длинной лентой отзывов."""
    if n <= 0 or n >= len(rows):
        return rows
    longs = [row for row in rows if len(row.get("reviews") or []) >= 90]
    shorts = [row for row in rows if len(row.get("reviews") or []) < 90]
    n_long = min(len(longs), max(1, n // 8))
    return shorts[: n - n_long] + longs[:n_long]


def load_params(path: str | Path | None) -> GenParams:
    if path is None:
        return GenParams()
    file_path = Path(path)
    if not file_path.exists():
        return GenParams()
    from cardgen.generate import params_from_dict

    return params_from_dict(load_json(file_path))
