"""Стоимость 1000 карточек и объём, при котором дешевле внешний API.

Цены условные, из условия задачи:
- аренда ВМ с T4 — 65 ₽/час, платим за время, пока машина включена;
- API — 20 ₽ за 1 млн входных токенов и 60 ₽ за 1 млн выходных.
"""

from __future__ import annotations

import math

GPU_RUB_PER_HOUR = 65.0
API_RUB_PER_MILLION_INPUT = 20.0
API_RUB_PER_MILLION_OUTPUT = 60.0
# Холодный старт vLLM, если его не измерили по логу. На T4 загрузка 1.5B обычно 1–3 минуты.
DEFAULT_STARTUP_SEC = 180.0
HOURS_IN_MONTH = 24 * 30


def cost_vllm_per_1000(elapsed_sec: float, n_cards: int) -> float:
    """(время_прогона_сек / 3600 × 65) / число_карточек × 1000."""
    if n_cards <= 0:
        raise ValueError("n_cards must be positive")
    return (elapsed_sec / 3600 * GPU_RUB_PER_HOUR) / n_cards * 1000


def cost_api_per_1000(input_tokens: int, output_tokens: int, n_cards: int) -> float:
    """(вход × 20 + выход × 60) / 1_000_000 / число_карточек × 1000."""
    if n_cards <= 0:
        raise ValueError("n_cards must be positive")
    spent = (
        input_tokens * API_RUB_PER_MILLION_INPUT
        + output_tokens * API_RUB_PER_MILLION_OUTPUT
    ) / 1_000_000
    return spent / n_cards * 1000


def monthly_costs(cards_per_month: int, per_1000_vllm: float, per_1000_api: float) -> dict[str, float]:
    """Линейная цена, если ВМ включена ровно на время генерации и простоя нет."""
    thousands = cards_per_month / 1000
    return {
        "vllm": thousands * per_1000_vllm,
        "api": thousands * per_1000_api,
    }


def break_even_with_startup(
    per_1000_vllm: float,
    per_1000_api: float,
    startup_sec: float = DEFAULT_STARTUP_SEC,
) -> float | None:
    """Месячный объём, ниже которого API дешевле, если ВМ поднимают под один прогон.

    Под один запуск платим холодный старт плюс само время генерации:
        65 × T_старт + N/1000 × C_vllm  против  N/1000 × C_api.
    Пока N меньше найденной границы, фиксированный старт дороже разницы в цене карточки.
    Если свой сервер дороже и на карточку, граница не существует: API дешевле всегда.
    """
    gap = per_1000_api - per_1000_vllm
    if gap <= 0:
        return None
    startup_rub = GPU_RUB_PER_HOUR * (startup_sec / 3600)
    return startup_rub / gap * 1000


def session_cost(
    n_cards: int,
    throughput: float,
    sessions: int,
    startup_sec: float = DEFAULT_STARTUP_SEC,
    *,
    round_up_hour: bool = False,
) -> float:
    """Сколько стоит разложить месячный объём на несколько включений ВМ.

    В каждое включение входит холодный старт и обработка своей доли карточек.
    При помесячной почасовой тарификации незавершённый час всё равно оплачивается.
    """
    if sessions <= 0 or throughput <= 0 or n_cards < 0:
        raise ValueError("некорректный объём сессии")
    seconds = n_cards / sessions / throughput + startup_sec
    hours = seconds / 3600
    if round_up_hour:
        hours = math.ceil(hours - 1e-9)
    return sessions * hours * GPU_RUB_PER_HOUR


def break_even_always_on(per_1000_api: float, hours_in_month: float = HOURS_IN_MONTH) -> float | None:
    """Объём, ниже которого API дешевле аренды ВМ на весь месяц без выключения.

    Месяц стоит 65 × часы независимо от числа карточек. Свой сервер окупает
    эту аренду, когда счёт API за тот же объём становится больше аренды.
    """
    if per_1000_api <= 0:
        return None
    return GPU_RUB_PER_HOUR * hours_in_month / per_1000_api * 1000
