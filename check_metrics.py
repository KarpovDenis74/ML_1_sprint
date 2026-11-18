"""Пересчёт метрик приёмки из сохранённого прогона.

Доля валидных считается заново по правилам витрины, скорость и стоимость —
по формулам из условия, из времени и токенов этого прогона.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cardgen.cost import (
    break_even_always_on,
    break_even_with_startup,
    cost_api_per_1000,
    cost_vllm_per_1000,
    monthly_costs,
)
from cardgen.io_utils import load_json, load_jsonl, write_json
from cardgen.metrics import build_report, evaluate_predictions, summarize_details
from cardgen.report_md import render_readme
from cardgen.runner import ROOT

PLAN_VOLUME = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Метрики прогона и отчёт")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark.jsonl")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs" / "predictions.jsonl")
    parser.add_argument("--details", type=Path, default=ROOT / "outputs" / "details.jsonl")
    parser.add_argument("--run-meta", type=Path, default=ROOT / "outputs" / "run_meta.json")
    parser.add_argument("--api", type=Path, default=ROOT / "outputs" / "api_comparison.json")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "report.json")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Проверить формулы стоимости на примере из условия и выйти",
    )
    return parser.parse_args()


def self_check() -> None:
    vllm = cost_vllm_per_1000(180, 300)
    api = cost_api_per_1000(34_000, 6_000, 25)
    if abs(vllm - 10.8333) > 0.01:
        raise SystemExit(f"формула ВМ дала {vllm}, ожидалось около 10.83")
    if abs(api - 41.6) > 0.01:
        raise SystemExit(f"формула API дала {api}, ожидалось 41.6")
    print(f"формулы сходятся с примером из условия: ВМ {vllm:.2f} ₽, API {api:.1f} ₽")


def assemble(args: argparse.Namespace) -> dict:
    products = load_jsonl(args.benchmark)
    predictions = load_jsonl(args.predictions)
    details = load_jsonl(args.details)
    meta = load_json(args.run_meta)
    if len(details) != len(products):
        raise SystemExit(
            f"details.jsonl содержит {len(details)} строк, во входе {len(products)}"
        )
    quality = evaluate_predictions(products, predictions)
    totals = summarize_details(details)
    api = load_json(args.api) if args.api.exists() else None
    report = build_report(
        model=meta["model"],
        concurrency=meta["concurrency"],
        params=meta["params"],
        elapsed_sec=meta["elapsed_sec"],
        n=len(products),
        valid=quality["valid"],
        errors_by_reason=quality["errors_by_reason"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        avg_latency=totals["avg_latency"],
        avg_attempts=totals["avg_attempts"],
        api_comparison=api,
    )
    # Числа metrics должны получаться из полей run. Проверяем это явно.
    run = report["run"]
    metrics = report["metrics"]
    throughput = run["n"] / run["elapsed_sec"]
    if abs(metrics["throughput"] - round(throughput, 4)) > 1e-9:
        raise SystemExit("throughput в metrics не сходится с n / elapsed_sec")
    expected_cost = round(cost_vllm_per_1000(run["elapsed_sec"], run["n"]), 4)
    if abs(metrics["cost_vllm_per_1000"] - expected_cost) > 1e-9:
        raise SystemExit("стоимость ВМ в metrics не сходится с формулой")
    if run["input_tokens"] != totals["input_tokens"] or run["output_tokens"] != totals["output_tokens"]:
        raise SystemExit("токены в run не совпали с суммой details")
    return report


def print_decision(report: dict) -> None:
    metrics = report["metrics"]
    api = report["api_comparison"]
    per_vllm = metrics["cost_vllm_per_1000"]
    per_api = api.get("cost_per_1000") or 0.0
    print(f"валидных: {metrics['valid']}/{report['run']['n']} ({metrics['valid_rate']:.1%})")
    print(f"брак: {metrics['errors_by_reason']}")
    print(
        f"скорость: {metrics['throughput']:.3f} карт/с, "
        f"latency {report['run']['avg_latency']:.2f} с, "
        f"300 карточек за {metrics['time_for_300_sec']} с"
    )
    print(f"свой сервер: {per_vllm:.2f} ₽/1000")
    if api.get("n"):
        print(
            f"API {api['model']}: валидных {api['valid_rate']:.1%}, "
            f"{per_api:.2f} ₽/1000 на {api['n']} карточках"
        )
        plan = monthly_costs(PLAN_VOLUME, per_vllm, per_api)
        startup = break_even_with_startup(per_vllm, per_api)
        always_on = break_even_always_on(per_api)
        print(
            f"план {PLAN_VOLUME} карточек/мес: свой сервер {plan['vllm']:.0f} ₽, "
            f"API {plan['api']:.0f} ₽"
        )
        if startup is None:
            print("при оплате только рабочего времени API не дешевле ни на каком объёме")
        else:
            print(f"если поднимать ВМ под прогон, API дешевле при объёме ниже {startup:.0f} карточек/мес")
        if always_on is not None:
            print(f"если держать ВМ весь месяц, API дешевле ниже {always_on:.0f} карточек/мес")
    gates = [
        ("валидность ≥ 80%", metrics["valid_rate"] >= 0.8),
        ("300 карточек ≤ 8 мин", (metrics["time_for_300_sec"] or 1e9) <= 480),
        ("выбранный вариант ≤ 25 ₽/1000", _chosen_cost(per_vllm, per_api) <= 25),
    ]
    for label, ok in gates:
        print(("пройден" if ok else "НЕ ПРОЙДЕН") + f": {label}")


def _chosen_cost(per_vllm: float, per_api: float) -> float:
    if per_api and per_api < per_vllm:
        return per_api
    return per_vllm


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    report = assemble(args)
    write_json(args.report, report)
    print_decision(report)
    print(f"записан {args.report}")
    if args.write_readme:
        text = render_readme(report)
        args.readme.write_text(text, encoding="utf-8")
        print(f"записан {args.readme}")
    if report["metrics"]["valid_rate"] < 0.8:
        sys.exit(2)


if __name__ == "__main__":
    main()
