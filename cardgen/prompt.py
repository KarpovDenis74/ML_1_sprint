"""Сборка промпта и укладка отзывов в бюджет 1500 токенов."""

from __future__ import annotations

import json
from typing import Any

# Вход карточки вместе с шаблоном чата. Ответ в этот бюджет не входит.
MAX_INPUT_TOKENS = 1500

# Шаблонные приставки, которыми в датасете разбавлены одни и те же мнения.
# Более длинные — раньше, чтобы «пользуюсь пару недель.» не отрезалось частично.
_PREFIXES = tuple(
    sorted(
        (
            "что могу сказать:",
            "коротко:",
            "в целом:",
            "в целом ",
            "заказывал на подарок.",
            "купил на замену старому.",
            "пользуюсь каждый день.",
            "пользуюсь пару недель.",
            "пользуюсь месяц.",
            "брал для себя.",
        ),
        key=len,
        reverse=True,
    )
)

EXAMPLE_CARD = {
    "product_id": "P-0000",
    "description": (
        "Беспроводные внутриканальные наушники с Bluetooth 5.3 и защитой IPX4 "
        "рассчитаны на спорт и каждый день. По отзывам они держат заряд до суток "
        "и не выпадают на пробежке. Покупателям нравится чистый звук, хотя на морозе "
        "батарея садится заметно быстрее."
    ),
    "pros": ["Чистый звук", "Плотная посадка", "До суток без подзарядки"],
    "cons": ["Быстро садятся на морозе", "На ярком солнце корпус нагревается"],
    "tags": ["наушники", "беспроводные", "спорт", "Bluetooth 5.3", "IPX4"],
}

SYSTEM_PROMPT = f"""Ты составляешь карточку товара для витрины маркетплейса. Верни только один json-объект и ничего больше: без Markdown, без пояснений, без текста до и после. Первая строка ответа начинается с {{.

Ключи ровно эти: product_id, description, pros, cons, tags.
- product_id — скопируй из запроса символ в символ. В образце стоит чужой id, его копировать нельзя.
- description — ровно 3 предложения на русском. Каждое заканчивается точкой. Без вопросов и восклицаний. Объём как в образце: примерно 250–450 символов, не одно короткое предложение и не длинный список.
- pros — ровно 3 разные короткие фразы без точки в конце.
- cons — ровно 2 разные короткие фразы без точки в конце.
- tags — ровно 5 разных коротких тегов без точки в конце. Не повторяй одно слово разными окончаниями.

Не выдумывай свойства, которых нет в характеристиках и отзывах. Число после знака × — сколько отзывов так сказали, в текст карточки его не копируй.

Образец объёма и структуры:
{json.dumps(EXAMPLE_CARD, ensure_ascii=False, indent=2)}
"""


def normalize_review(text: str) -> str:
    """Убирает шаблонную приставку и схлопывает пробелы. Пустая строка — отзыва не осталось."""
    cleaned = " ".join(text.split())
    while cleaned:
        folded = cleaned.casefold()
        removed = False
        for prefix in _PREFIXES:
            if folded.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" \t:;,.–—-")
                removed = True
                break
        if not removed:
            break
    return cleaned.strip()


def dedupe_reviews(reviews: list[str]) -> list[tuple[str, int]]:
    """Уникальные мнения в порядке убывания частоты, при равенстве — как в исходном списке."""
    order: list[str] = []
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    for raw in reviews:
        core = normalize_review(raw)
        if not core:
            continue
        key = core.casefold()
        if key not in counts:
            counts[key] = 0
            order.append(key)
            display[key] = core
        counts[key] += 1
    ranked = sorted(order, key=lambda key: (-counts[key], order.index(key)))
    return [(display[key], counts[key]) for key in ranked]


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Число токенов входа так, как его увидит модель: шаблон чата плюс запрос на ответ."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except ImportError:
        # Старый jinja2 не умеет шаблон Qwen. Ручная сборка совпадает с ним по токенам.
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        rendered += "<|im_start|>assistant\n"
        encoded = tokenizer.encode(rendered)
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    return len(encoded)


def _format_attributes(attributes: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in attributes.items())


def _format_reviews(items: list[tuple[str, int]], with_counts: bool) -> str:
    if not items:
        return "Отзывов нет."
    lines = []
    for text, count in items:
        if with_counts and count > 1:
            lines.append(f"- {text} ×{count}")
        else:
            lines.append(f"- {text}")
    return "\n".join(lines)


def build_messages(
    product: dict[str, Any],
    reviews: list[tuple[str, int]],
    *,
    with_counts: bool,
) -> list[dict[str, str]]:
    product_id = product["product_id"]
    user = (
        f"product_id: {product_id}\n"
        f"Название: {product['title']}\n"
        f"Категория: {product['category']}\n"
        f"Характеристики:\n{_format_attributes(product['attributes'])}\n\n"
        f"Отзывы:\n{_format_reviews(reviews, with_counts)}\n\n"
        f'Верни json. product_id = "{product_id}". '
        "В description ровно 3 предложения. pros — 3 пункта, cons — 2 пункта, tags — 5 пунктов."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _select_within_limit(
    tokenizer: Any,
    product: dict[str, Any],
    unique: list[tuple[str, int]],
    max_input_tokens: int,
) -> tuple[list[tuple[str, int]], list[dict[str, str]], int]:
    """Оставляет самые частые мнения и хвост редких, пока промпт не влезет в бюджет.

    Редкие формулировки нужны потому, что минусы в этих отзывах встречаются реже плюсов.
    Если оставить только топ по частоте, блок минусов останется без опоры.
    """

    def pack(items: list[tuple[str, int]]) -> tuple[list[dict[str, str]], int]:
        messages = build_messages(product, items, with_counts=False)
        return messages, count_prompt_tokens(tokenizer, messages)

    def mix(count: int) -> list[tuple[str, int]]:
        if count >= len(unique) or count <= 0:
            return unique[:count]
        head_count = max(1, (count * 7) // 10)
        head = unique[:head_count]
        tail_count = count - len(head)
        tail = unique[-tail_count:] if tail_count else []
        seen = {text.casefold() for text, _ in head}
        merged = list(head)
        for item in tail:
            if item[0].casefold() not in seen:
                merged.append(item)
                seen.add(item[0].casefold())
        return merged

    low, high = 0, len(unique)
    best_count = 0
    best_messages, best_tokens = pack([])
    while low <= high:
        mid = (low + high) // 2
        messages, tokens = pack(mix(mid))
        if tokens <= max_input_tokens:
            best_count = mid
            best_messages, best_tokens = messages, tokens
            low = mid + 1
        else:
            high = mid - 1
    return mix(best_count), best_messages, best_tokens


def fit_product(
    tokenizer: Any,
    product: dict[str, Any],
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Собирает вход карточки не длиннее MAX_INPUT_TOKENS.

    Сначала снимает шаблонные приставки и повторы. Если уникальные мнения влезают
    все, частота сохраняется в тексте. Если нет — частота убирается, затем
    отбрасываются лишние отзывы.
    """
    unique = dedupe_reviews(list(product.get("reviews") or []))
    info = {
        "reviews_total": len(product.get("reviews") or []),
        "reviews_unique": len(unique),
        "reviews_used": len(unique),
        "with_counts": True,
        "truncated": False,
    }

    messages = build_messages(product, unique, with_counts=True)
    tokens = count_prompt_tokens(tokenizer, messages)
    if tokens <= max_input_tokens:
        info["tokens"] = tokens
        return messages, info

    messages = build_messages(product, unique, with_counts=False)
    tokens = count_prompt_tokens(tokenizer, messages)
    info["with_counts"] = False
    if tokens <= max_input_tokens:
        info["tokens"] = tokens
        return messages, info

    chosen, messages, tokens = _select_within_limit(
        tokenizer, product, unique, max_input_tokens
    )
    info["reviews_used"] = len(chosen)
    info["truncated"] = len(chosen) < len(unique)
    info["tokens"] = tokens
    if tokens > max_input_tokens:
        # Даже карточка без отзывов не влезла: дальше резать нечего, генерацию не роняем.
        info["over_budget"] = True
    return messages, info


def repair_message(product_id: str, issues: list[Any]) -> str:
    lines = [
        "Предыдущий ответ витрина не приняла. Верни исправленный json-объект и больше ничего.",
        "Не копируй образец и не меняй удачные поля без нужды.",
        f'product_id должен быть ровно "{product_id}".',
    ]
    for issue in issues:
        lines.append(f"- {issue.message}")
    lines.append(
        "Напоминание: description — ровно 3 предложения с точкой в конце, объём как в образце; "
        "pros — 3 пункта, cons — 2 пункта, tags — 5 пунктов."
    )
    return "\n".join(lines)
