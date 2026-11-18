"""Проверки формата, бюджета входа, повтора запроса и формул стоимости. Сервер не нужен."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from cardgen.cost import (
    break_even_always_on,
    break_even_with_startup,
    cost_api_per_1000,
    cost_vllm_per_1000,
)
from cardgen.generate import CardGenerator, GenParams
from cardgen.metrics import build_report
from cardgen.prompt import EXAMPLE_CARD, dedupe_reviews, fit_product
from cardgen.runner import ROOT
from cardgen.validate import count_sentences, extract_json, validate_card
from run_batch import choose_concurrency

OFFICIAL = {
    "product_id": "P-1001",
    "description": (
        "Компактные внутриканальные наушники с Bluetooth 5.3 и защитой IPX4. "
        "Держат заряд до 24 часов и надёжно сидят при активном движении. "
        "Подойдут для спорта и повседневного использования."
    ),
    "pros": ["Чистый звук", "Удобная посадка", "Долгое время работы"],
    "cons": ["Быстро разряжаются на морозе"],
    "tags": ["наушники", "беспроводные", "спорт", "IPX4", "Bluetooth 5.3"],
}


class FormatTests(unittest.TestCase):
    def test_decimal_point_is_not_a_sentence_boundary(self):
        text = "Экран 6.1 дюйма и Bluetooth 5.3. Вес 0.5 кг. Хватает на день."
        self.assertEqual(count_sentences(text), 3)

    def test_official_example_is_valid(self):
        self.assertEqual(validate_card(OFFICIAL, "P-1001"), [])
        self.assertGreaterEqual(len(OFFICIAL["description"]), 120)
        self.assertEqual(count_sentences(OFFICIAL["description"]), 3)

    def test_prompt_example_is_valid(self):
        self.assertEqual(validate_card(EXAMPLE_CARD, "P-0000"), [])

    def test_rejects_wrong_shape(self):
        card = {
            "product_id": "P-2",
            "description": "Слишком коротко.",
            "pros": "не список",
            "cons": ["один"],
            "tags": ["a", "b"],
        }
        codes = [issue.code for issue in validate_card(card, "P-1")]
        self.assertIn("product_id", codes)
        self.assertIn("длина description", codes)
        self.assertIn("предложений", codes)
        self.assertIn("pros", codes)
        self.assertIn("tags", codes)

    def test_extracts_json_from_markdown_and_preamble(self):
        raw = 'Вот карточка:\n```json\n{"product_id": "P-1", "description": "x"}\n```'
        self.assertEqual(extract_json(raw)["product_id"], "P-1")


class DedupeTests(unittest.TestCase):
    def test_boilerplate_collapses_to_one_opinion(self):
        reviews = [
            "Коротко: Звук чистый, бас глубокий",
            "Пользуюсь месяц. Звук чистый, бас глубокий",
            "В целом Звук чистый, бас глубокий",
            "На морозе быстро садятся",
        ]
        unique = dedupe_reviews(reviews)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0], ("Звук чистый, бас глубокий", 3))
        self.assertEqual(unique[1][0], "На морозе быстро садятся")

    def test_long_catalog_keeps_rare_complaints(self):
        path = ROOT / "data" / "benchmark.jsonl"
        if not path.exists():
            self.skipTest("нет benchmark.jsonl")
        import json

        target = None
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["product_id"] == "P-1262":
                target = row
                break
        self.assertIsNotNone(target)
        texts = [text for text, _ in dedupe_reviews(target["reviews"])]
        self.assertTrue(any("дешевовато" in text for text in texts))
        self.assertTrue(any("Шумит" in text for text in texts))
        self.assertLess(len(texts), 30)
        self.assertGreater(len(target["reviews"]), 200)


class BudgetTests(unittest.TestCase):
    def test_every_benchmark_card_fits(self):
        tokenizer_dir = ROOT / "data" / "tokenizer"
        dataset = ROOT / "data" / "benchmark.jsonl"
        if not (tokenizer_dir / "tokenizer.json").exists() or not dataset.exists():
            self.skipTest("нет токенайзера или датасета")
        from transformers import AutoTokenizer

        from cardgen.io_utils import load_jsonl

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        rows = load_jsonl(dataset)
        worst = 0
        for row in rows:
            _, info = fit_product(tokenizer, row)
            worst = max(worst, info["tokens"])
            self.assertLessEqual(info["tokens"], 1500, row["product_id"])
            self.assertFalse(info["truncated"], row["product_id"])
            self.assertFalse(info.get("over_budget"), row["product_id"])
            if len(row["reviews"]) >= 90:
                self.assertEqual(info["reviews_used"], info["reviews_unique"])
                self.assertGreaterEqual(info["reviews_unique"], 8)
        self.assertGreater(worst, 700)
        self.assertLessEqual(worst, 1500)


class CostTests(unittest.TestCase):
    def test_matches_the_task_example(self):
        self.assertAlmostEqual(cost_vllm_per_1000(180, 300), 10.8333, places=2)
        self.assertAlmostEqual(cost_api_per_1000(34_000, 6_000, 25), 41.6, places=2)

    def test_break_even(self):
        per_vllm = cost_vllm_per_1000(180, 300)
        per_api = cost_api_per_1000(34_000, 6_000, 25)
        startup = break_even_with_startup(per_vllm, per_api, startup_sec=180)
        self.assertAlmostEqual(startup, 105.6, places=0)
        self.assertEqual(break_even_with_startup(50, 10), None)
        self.assertAlmostEqual(break_even_always_on(41.6), 1_125_000, places=0)


class ReportTests(unittest.TestCase):
    def test_metrics_are_derived_from_run(self):
        report = build_report(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            concurrency=16,
            params={"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.1, "max_tokens": 350},
            elapsed_sec=180,
            n=300,
            valid=250,
            errors_by_reason={"предложений": 10},
            input_tokens=1000,
            output_tokens=500,
            avg_latency=2.5,
            avg_attempts=1.2,
        )
        run = report["run"]
        metrics = report["metrics"]
        self.assertEqual(metrics["valid"], 250)
        self.assertAlmostEqual(metrics["throughput"], round(300 / run["elapsed_sec"], 4))
        self.assertAlmostEqual(metrics["time_for_300_sec"], run["elapsed_sec"])
        self.assertAlmostEqual(metrics["cost_vllm_per_1000"], round(cost_vllm_per_1000(180, 300), 4))


class RetryTests(unittest.TestCase):
    def test_second_attempt_fixes_the_format(self):
        product = {
            "product_id": "P-1",
            "title": "Наушники",
            "category": "Электроника",
            "attributes": {"бренд": "X"},
            "reviews": ["Звук чистый", "На морозе садятся"],
        }
        bad = '{"product_id": "P-1", "description": "Коротко.", "pros": ["а"], "cons": ["б"], "tags": ["в"]}'
        good_card = dict(EXAMPLE_CARD)
        good_card["product_id"] = "P-1"
        import json

        good = json.dumps(good_card, ensure_ascii=False)

        class Completions:
            def __init__(self):
                self.calls = 0

            async def create(self, **kwargs):
                self.calls += 1
                text = bad if self.calls == 1 else good
                message = SimpleNamespace(content=text)
                choice = SimpleNamespace(message=message, finish_reason="stop")
                usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40)
                return SimpleNamespace(choices=[choice], usage=usage)

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                return [0] * 100

        generator = CardGenerator(
            client,
            Tokenizer(),
            "test-model",
            GenParams(max_attempts=2, json_mode="off"),
        )
        result = asyncio.run(generator.generate_one(product))
        self.assertTrue(result["valid"], result["error_messages"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["input_tokens"], 200)
        self.assertEqual(result["output_tokens"], 80)
        self.assertEqual(completions.calls, 2)


class ConcurrencyTests(unittest.TestCase):
    def test_stops_when_throughput_plateaus_or_quality_drops(self):
        chosen = choose_concurrency(
            [
                {"concurrency": 1, "throughput": 0.4, "valid_rate": 0.9},
                {"concurrency": 4, "throughput": 1.2, "valid_rate": 0.9},
                {"concurrency": 16, "throughput": 1.26, "valid_rate": 0.88},
                {"concurrency": 32, "throughput": 1.5, "valid_rate": 0.7},
            ]
        )
        self.assertEqual(chosen, 4)


if __name__ == "__main__":
    unittest.main()
