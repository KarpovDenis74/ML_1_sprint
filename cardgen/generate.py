"""Один товар → карточка. Повтор запроса, если витрина не приняла формат."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

from cardgen.prompt import fit_product, repair_message
from cardgen.validate import extract_json, validate_card

# Схема не ограничивает длину description: модель не умеет считать символы,
# а грамматический minLength на сотнях символов часто зацикливает декодирование.
CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "product_id": {"type": "string"},
        "description": {"type": "string"},
        "pros": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1},
        },
        "cons": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string", "minLength": 1},
        },
        "tags": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": ["product_id", "description", "pros", "cons", "tags"],
}

_PROBE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


@dataclass
class GenParams:
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int | None = 50
    repetition_penalty: float = 1.1
    max_tokens: int = 350
    seed: int | None = 1
    max_attempts: int = 2
    # guided — грамматический JSON vLLM; object — response_format json_object; off — только промпт.
    json_mode: str = "guided"
    # top_k и repetition_penalty есть у vLLM и отсутствуют у большинства внешних API.
    send_vllm_extras: bool = True

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k is not None else "unlimited",
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
        }


def params_from_dict(data: dict[str, Any], base: GenParams | None = None) -> GenParams:
    params = replace(base) if base is not None else GenParams()
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
        "seed",
        "max_attempts",
        "json_mode",
        "send_vllm_extras",
    ):
        if key not in data:
            continue
        value = data[key]
        if key == "top_k" and value == "unlimited":
            value = None
        setattr(params, key, value)
    return params


class CardGenerator:
    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        model: str,
        params: GenParams | None = None,
        *,
        handle_http_errors: bool = False,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.model = model
        self.params = params or GenParams()
        self.handle_http_errors = handle_http_errors
        self.input_budget = 1500
        self.http_errors: dict[str, int] = {"429": 0, "400": 0, "5xx": 0, "timeout": 0, "other": 0}

    async def probe_json_mode(self) -> str:
        """Проверяет, что выбранный режим JSON сервер принимает, и откатывается, если нет."""
        if self.params.json_mode == "off":
            print("режим json: off", flush=True)
            return "off"
        messages = [{"role": "user", "content": 'Верни json {"ok": true}'}]
        try:
            await self._complete(messages, mode=self.params.json_mode, schema=_PROBE_SCHEMA, max_tokens=30)
        except Exception as exc:
            # 400 и 5xx здесь значат, что схема не принята. Обрыв связи и 429 — это не схема.
            if _error_kind(exc) not in {"400", "5xx"}:
                raise
            print(f"режим json недоступен ({exc}), дальше без ограничения схемы", flush=True)
            self.params.json_mode = "off"
        print(f"режим json: {self.params.json_mode}", flush=True)
        return self.params.json_mode

    async def generate_one(self, product: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return await self._generate(product, started)
        except Exception as exc:  # noqa: BLE001 — карточка не должна ронять весь пакет
            return _empty_result(
                product,
                latency=time.perf_counter() - started,
                errors=["api"],
                messages=[str(exc)],
            )

    async def _generate(self, product: dict[str, Any], started: float) -> dict[str, Any]:
        product_id = product["product_id"]
        budget_limit = self.input_budget
        messages, budget = fit_product(self.tokenizer, product, budget_limit)
        attempts = 0
        input_tokens = 0
        output_tokens = 0
        card: dict[str, Any] | None = None
        issues: list[Any] = []
        finish_reasons: list[str] = []
        last_raw = ""

        while attempts < self.params.max_attempts:
            attempts += 1
            try:
                response = await self._complete(messages, mode=self.params.json_mode, schema=CARD_SCHEMA)
            except Exception as exc:
                kind = _error_kind(exc)
                issues = [_ApiIssue(kind, str(exc))]
                if kind == "400" and _is_context_length(exc) and budget_limit > 400:
                    budget_limit = max(400, budget_limit // 2)
                    messages, budget = fit_product(self.tokenizer, product, budget_limit)
                    attempts -= 1
                    continue
                if kind == "400":
                    break
                continue

            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            choice = response.choices[0]
            finish_reasons.append(getattr(choice, "finish_reason", None) or "")
            last_raw = _message_text(choice.message)
            try:
                card = extract_json(last_raw)
            except ValueError as exc:
                issues = [_ApiIssue("json", str(exc))]
                messages = _with_repair(messages, last_raw, repair_message(product_id, issues))
                continue

            issues = validate_card(card, product_id)
            if not issues:
                break
            messages = _with_repair(messages, last_raw, repair_message(product_id, issues))

        return {
            "product_id": product_id,
            "card": card,
            "valid": bool(card is not None and not issues),
            "errors": [issue.code for issue in issues],
            "error_messages": [issue.message for issue in issues],
            "attempts": attempts,
            "latency_sec": time.perf_counter() - started,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "first_prompt_tokens": budget["tokens"],
            "reviews_total": budget["reviews_total"],
            "reviews_unique": budget["reviews_unique"],
            "reviews_used": budget["reviews_used"],
            "truncated_input": budget["truncated"],
            "finish_reasons": finish_reasons,
            "last_raw": last_raw,
        }

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str,
        schema: dict[str, Any],
        max_tokens: int | None = None,
    ) -> Any:
        delay = 1.0
        transport_tries = 0
        drop_extras = not self.params.send_vllm_extras
        active_mode = mode
        while True:
            kwargs = _request_kwargs(
                self.model,
                messages,
                self.params,
                mode=active_mode,
                schema=schema,
                max_tokens=max_tokens,
                drop_extras=drop_extras,
            )
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                kind = _error_kind(exc)
                self.http_errors[kind] = self.http_errors.get(kind, 0) + 1
                if kind == "400":
                    message = str(exc).lower()
                    if _is_context_length(exc):
                        raise
                    if not drop_extras and _looks_like_unsupported_param(message):
                        drop_extras = True
                        self.params.send_vllm_extras = False
                        continue
                    if active_mode == "guided":
                        active_mode = "object"
                        self.params.json_mode = "object"
                        continue
                    if active_mode == "object":
                        active_mode = "off"
                        self.params.json_mode = "off"
                        continue
                    raise
                if not _is_transient(exc) and not self.handle_http_errors:
                    raise
                transport_tries += 1
                if transport_tries > 5:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


class _ApiIssue:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


def _request_kwargs(
    model: str,
    messages: list[dict[str, str]],
    params: GenParams,
    *,
    mode: str,
    schema: dict[str, Any],
    max_tokens: int | None,
    drop_extras: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": max_tokens if max_tokens is not None else params.max_tokens,
    }
    if params.seed is not None and not drop_extras:
        kwargs["seed"] = params.seed
    extra: dict[str, Any] = {}
    if not drop_extras:
        if params.top_k is not None:
            extra["top_k"] = params.top_k
        if params.repetition_penalty is not None:
            extra["repetition_penalty"] = params.repetition_penalty
        if mode == "guided":
            extra["guided_json"] = schema
    if mode == "object":
        kwargs["response_format"] = {"type": "json_object"}
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None) or ""
    if content.strip():
        return content
    # Часть моделей OpenRouter кладёт текст в reasoning, а content оставляет пустым.
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if isinstance(value, str) and value.strip():
            return value
    return content


def _with_repair(
    messages: list[dict[str, str]], raw: str, repair: str
) -> list[dict[str, str]]:
    updated = list(messages)
    # В диалоге оставляем последнюю неудачную попытку, а не всю цепочку: так повтор
    # не раздувает контекст сверх max-model-len.
    if len(updated) > 2:
        updated = updated[:2]
    updated.append({"role": "assistant", "content": raw[:2000]})
    updated.append({"role": "user", "content": repair})
    return updated


def _empty_result(
    product: dict[str, Any],
    *,
    latency: float,
    errors: list[str],
    messages: list[str],
) -> dict[str, Any]:
    return {
        "product_id": product["product_id"],
        "card": None,
        "valid": False,
        "errors": errors,
        "error_messages": messages,
        "attempts": 0,
        "latency_sec": latency,
        "input_tokens": 0,
        "output_tokens": 0,
        "first_prompt_tokens": 0,
        "reviews_total": len(product.get("reviews") or []),
        "reviews_unique": 0,
        "reviews_used": 0,
        "truncated_input": False,
        "finish_reasons": [],
        "last_raw": "",
    }


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _error_kind(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    status = _status_code(exc)
    if status == 429 or "ratelimit" in name:
        return "429"
    if status == 400 or "badrequest" in name:
        return "400"
    if status is not None and status >= 500:
        return "5xx"
    if "connection" in name or "apiconnection" in name:
        return "other"
    return "other"


def _looks_like_unsupported_param(message: str) -> bool:
    return any(
        token in message
        for token in ("top_k", "repetition", "extra", "seed", "unknown", "unrecognized", "unexpected")
    )


def _is_context_length(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in ("context length", "maximum context", "too many tokens", "context window", "input is too long")
    )


def _is_transient(exc: Exception) -> bool:
    if _error_kind(exc) in {"429", "5xx", "timeout"}:
        return True
    name = type(exc).__name__.lower()
    return "connection" in name or "timeout" in name


def public_card(result: dict[str, Any]) -> dict[str, Any]:
    """Карточка для predictions.jsonl: как вернула модель, либо пустая заготовка."""
    card = result.get("card")
    if isinstance(card, dict):
        return card
    return {
        "product_id": result["product_id"],
        "description": "",
        "pros": [],
        "cons": [],
        "tags": [],
    }
