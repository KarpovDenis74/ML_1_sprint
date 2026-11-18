"""Метрики приёмки: валидность, скорость, стоимость. Считаются из файлов прогона."""

from __future__ import annotations

from collections import Counter
from typing import Any

from cardgen.cost import cost_api_per_1000, cost_vllm_per_1000
from cardgen.validate import validate_card


def evaluate_predictions(
    products: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(products) != len(predictions):
        raise ValueError(
            f"число товаров ({len(products)}) и карточек ({len(predictions)}) не совпадает"
        )
    valid = 0
    reasons: Counter[str] = Counter()
    description_lengths: list[int] = []
    for product, card in zip(products, predictions):
        issues = validate_card(card, product["product_id"])
        if not issues:
            valid += 1
        else:
            for issue in issues:
                reasons[issue.code] += 1
        description = card.get("description") if isinstance(card, dict) else None
        if isinstance(description, str) and description:
            description_lengths.append(len(description))
    return {
        "valid": valid,
        "valid_rate": valid / len(products) if products else 0.0,
        "errors_by_reason": dict(reasons.most_common()),
        "avg_description_len": (
            sum(description_lengths) / len(description_lengths) if description_lengths else 0.0
        ),
    }


def summarize_details(details: list[dict[str, Any]]) -> dict[str, float]:
    n = len(details)
    if n == 0:
        return {"input_tokens": 0, "output_tokens": 0, "avg_latency": 0.0, "avg_attempts": 0.0}
    return {
        "input_tokens": int(sum(int(row.get("input_tokens") or 0) for row in details)),
        "output_tokens": int(sum(int(row.get("output_tokens") or 0) for row in details)),
        "avg_latency": sum(float(row.get("latency_sec") or 0) for row in details) / n,
        "avg_attempts": sum(float(row.get("attempts") or 0) for row in details) / n,
    }


def build_report(
    *,
    model: str,
    concurrency: int,
    params: dict[str, Any],
    elapsed_sec: float,
    n: int,
    valid: int,
    errors_by_reason: dict[str, int],
    input_tokens: int,
    output_tokens: int,
    avg_latency: float,
    avg_attempts: float,
    api_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает report.json. metrics выводятся из run теми же формулами, что в условии."""
    elapsed_sec = round(float(elapsed_sec), 3)
    run = {
        "model": model,
        "concurrency": concurrency,
        "params": params,
        "elapsed_sec": elapsed_sec,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "n": int(n),
        "avg_latency": round(float(avg_latency), 3),
        "avg_attempts": round(float(avg_attempts), 3),
    }
    throughput = n / elapsed_sec if elapsed_sec else 0.0
    metrics = {
        "valid": int(valid),
        "valid_rate": round(valid / n, 4) if n else 0.0,
        "throughput": round(throughput, 4),
        "time_for_300_sec": round(300 / throughput, 3) if throughput else None,
        "cost_vllm_per_1000": round(cost_vllm_per_1000(elapsed_sec, n), 4),
        "errors_by_reason": errors_by_reason,
    }
    api = api_comparison or {
        "model": "",
        "n": 0,
        "valid_rate": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_sec": 0.0,
        "cost_per_1000": 0.0,
    }
    return {"run": run, "metrics": metrics, "api_comparison": _public_api(api)}


def _public_api(api: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": api.get("model", ""),
        "n": int(api.get("n") or 0),
        "valid_rate": api.get("valid_rate") or 0.0,
        "input_tokens": int(api.get("input_tokens") or 0),
        "output_tokens": int(api.get("output_tokens") or 0),
        "elapsed_sec": api.get("elapsed_sec") or 0.0,
        "cost_per_1000": api.get("cost_per_1000") or 0.0,
    }


def api_comparison_from_run(
    model: str,
    products: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    details: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    quality = evaluate_predictions(products, predictions)
    totals = summarize_details(details)
    n = len(products)
    cost = cost_api_per_1000(totals["input_tokens"], totals["output_tokens"], n) if n else 0.0
    return {
        "model": model,
        "n": n,
        "valid_rate": round(quality["valid_rate"], 4),
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "elapsed_sec": round(elapsed_sec, 3),
        "cost_per_1000": round(cost, 4),
        "valid": quality["valid"],
        "errors_by_reason": quality["errors_by_reason"],
        "avg_latency": round(totals["avg_latency"], 3),
        "avg_description_len": round(quality["avg_description_len"], 1),
    }
