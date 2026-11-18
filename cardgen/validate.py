"""Проверка формата карточки. Смысл текста не оценивается — только структура витрины."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def count_sentences(text: str) -> int:
    """Число предложений в тексте. Точка внутри числа концом предложения не считается."""
    return len([s for s in re.split(r"[.!?]+(?!\d)", text) if s.strip()])


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели.

    Снимает ограждение ```json и игнорирует текст вокруг первого объекта.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            raise ValueError("в ответе нет JSON-объекта") from None
        try:
            value, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON не разбирается: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON должен быть объектом, а не списком или строкой")
    return value


def _list_issue(code: str, value: object, low: int, high: int) -> Issue | None:
    if not isinstance(value, list):
        return Issue(code, f"{code} должен быть массивом строк, а пришло {type(value).__name__}.")
    empty = [item for item in value if not isinstance(item, str) or not item.strip()]
    count = len(value)
    if empty:
        filled = count - len(empty)
        return Issue(
            code,
            f"{code}: нужно от {low} до {high} непустых строк, сейчас элементов {count}, из них непустых строк {filled}.",
        )
    if not low <= count <= high:
        return Issue(code, f"{code}: нужно от {low} до {high} пунктов, сейчас {count}.")
    return None


def validate_card(card: object, product_id: str) -> list[Issue]:
    """Возвращает список нарушений. Пустой список — карточка валидна."""
    if not isinstance(card, dict):
        return [Issue("json", "нужен JSON-объект.")]

    issues: list[Issue] = []
    required = ("product_id", "description", "pros", "cons", "tags")
    missing = [key for key in required if key not in card]
    if missing:
        issues.append(Issue("поля", "нет полей: " + ", ".join(missing) + "."))

    if "product_id" in card and card["product_id"] != product_id:
        issues.append(
            Issue(
                "product_id",
                f'product_id должен быть ровно "{product_id}", сейчас {card["product_id"]!r}.',
            )
        )

    if "description" in card:
        description = card["description"]
        if not isinstance(description, str):
            issues.append(Issue("длина description", "description должен быть строкой."))
        else:
            length = len(description)
            if not 120 <= length <= 700:
                issues.append(
                    Issue(
                        "длина description",
                        f"длина description {length} символов, нужно от 120 до 700. "
                        "Напиши ровно 3 полных предложения, по объёму как образец.",
                    )
                )
            sentences = count_sentences(description)
            if not 2 <= sentences <= 5:
                issues.append(
                    Issue(
                        "предложений",
                        f"в description {sentences} предложений, нужно от 2 до 5. "
                        "Напиши ровно 3 предложения, каждое заканчивается точкой. "
                        "Не ставь вопросительный и восклицательный знаки.",
                    )
                )

    for code, low, high in (("pros", 2, 5), ("cons", 1, 3), ("tags", 3, 8)):
        if code in card:
            issue = _list_issue(code, card[code], low, high)
            if issue:
                issues.append(issue)
    return issues


def is_valid(card: object, product_id: str) -> bool:
    return not validate_card(card, product_id)
