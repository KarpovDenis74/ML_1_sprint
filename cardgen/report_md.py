"""Текст отчёта. Числа подставляются из outputs/, пока прогона нет — остаётся метод."""

from __future__ import annotations

from cardgen.cost import (
    HOURS_IN_MONTH,
    GPU_RUB_PER_HOUR,
    break_even_always_on,
    break_even_with_startup,
    monthly_costs,
    session_cost,
)
from cardgen.io_utils import load_json
from cardgen.runner import ROOT

PLAN_VOLUME = 50_000
VLLM_COMMAND = """nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 \\
  --gpu-memory-utilization 0.9 --enforce-eager --port 8000 > vllm.log 2>&1 &"""


def _load(name: str) -> dict | None:
    path = ROOT / "outputs" / name
    if not path.exists():
        return None
    return load_json(path)


def _fmt(value, digits=1, suffix=""):
    if value is None:
        return "после прогона"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _pct(value) -> str:
    if value is None:
        return "после прогона"
    return f"{value * 100:.1f}%"


def _budget_paragraph() -> str:
    diagnostics = _load("diagnostics.json")
    if diagnostics and diagnostics.get("max_first_prompt_tokens"):
        return (
            f"В полном прогоне самый длинный вход — {diagnostics['max_first_prompt_tokens']} токенов, "
            f"карточек длиннее 1500 — {diagnostics.get('n_over_1500', 0)}, "
            f"с отброшенными отзывами — {diagnostics.get('n_truncated', 0)}. "
            f"Длинных товаров {diagnostics.get('n_long', 0)}: в среднем "
            f"{diagnostics.get('avg_unique_long', 0)} уникальных формулировок, "
            f"в промпт вошло {diagnostics.get('avg_reviews_used_long', 0)}."
        )
    dataset = ROOT / "data" / "benchmark.jsonl"
    tokenizer_dir = ROOT / "data" / "tokenizer"
    if not dataset.exists() or not (tokenizer_dir / "tokenizer.json").exists():
        return (
            "Бюджет проверяется токенайзером Qwen2.5 перед каждым запросом. "
            "Если уникальные отзывы всё же не влезают, остаются самые частые формулировки "
            "и часть редких: редкие чаще оказываются минусами."
        )
    try:
        from transformers import AutoTokenizer

        from cardgen.io_utils import load_jsonl
        from cardgen.prompt import fit_product

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        rows = load_jsonl(dataset)
        infos = [fit_product(tokenizer, row)[1] for row in rows]
        tokens = [info["tokens"] for info in infos]
        long = [info for row, info in zip(rows, infos) if len(row["reviews"]) >= 90]
        avg_unique = sum(info["reviews_unique"] for info in long) / len(long)
        avg_raw = sum(info["reviews_total"] for info in long) / len(long)
        return (
            f"На токенайзере Qwen2.5-1.5B-Instruct по всем {len(rows)} товарам benchmark.jsonl "
            f"вход получился от {min(tokens)} до {max(tokens)} токенов, ни один не превысил 1500 "
            f"и ни один не пришлось резать. У {len(long)} товаров с лентой от 90 отзывов "
            f"в сыром виде в среднем {avg_raw:.0f} отзывов, после очистки остаётся "
            f"около {avg_unique:.0f} разных формулировок, и в промпт входят все."
        )
    except Exception as exc:
        return f"Замер бюджета токенайзером не собрался ({exc}). На ВМ его повторит прогон."


def _tune_table(tune: dict | None) -> str:
    if not tune:
        return (
            "Сетка ещё не прогнана. Её считает `python tune_params.py` на 20 товарах dev.jsonl, "
            "по одному параметру от базовой точки. Базовая точка: temperature 0.3, top_p 0.9, "
            "top_k 50, repetition_penalty 1.1, max_tokens 350, json-режим guided.\n\n"
            "| параметр | значение | доля валидных | средняя длина description | выходные токены |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| — | — | после прогона | после прогона | после прогона |"
        )
    lines = [
        "Сетка на 20 товарах dev.jsonl. В каждой строке меняется один параметр, остальные стоят в базовой точке "
        "(temperature 0.3, top_p 0.9, top_k 50, repetition_penalty 1.1, max_tokens 350).",
        "",
        "| параметр | значение | доля валидных | средняя длина description | выходные токены | обрывы по max_tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in tune["rows"]:
        value = "без ограничения" if row["value"] == "unlimited" else row["value"]
        lines.append(
            f"| {row['parameter']} | {value} | {row['valid_rate'] * 100:.1f}% | "
            f"{row['avg_description_len']:.0f} | {row['avg_output_tokens']:.0f} | "
            f"{row['truncation_rate'] * 100:.0f}% |"
        )
    chosen = tune["chosen"]
    top_k = "без ограничения" if chosen["top_k"] is None else chosen["top_k"]
    confirmation = tune.get("confirmation") or {}
    lines.append("")
    lines.append(
        f"Выбрано: temperature {chosen['temperature']}, top_p {chosen['top_p']}, "
        f"top_k {top_k}, repetition_penalty {chosen['repetition_penalty']}, "
        f"max_tokens {chosen['max_tokens']}, json-режим {chosen['json_mode']}, "
        f"попыток до {chosen['max_attempts']}."
    )
    lines.append(
        "Правило выбора: у параметра берётся значение с лучшей долей валидных; "
        "если соседние значения отстают не больше чем на 2 процентных пункта, остаётся то, "
        "которое тратит меньше выходных токенов. Для max_tokens дополнительно отбрасываются "
        "значения, на которых JSON обрывается по лимиту, и из оставшихся берётся меньший лимит. "
        "Для json-режима при той же доле валидных берётся более быстрый."
    )
    if confirmation:
        lines.append(
            f"Эта комбинация на всех 40 товарах dev дала долю валидных "
            f"{confirmation['valid_rate'] * 100:.1f}% "
            f"при средней длине описания {confirmation['avg_description_len']:.0f} символов "
            f"и {confirmation['avg_output_tokens']:.0f} выходных токенах на карточку "
            f"(с учётом повторов). Брак: {confirmation.get('errors_by_reason') or 'нет'}."
        )
    return "\n".join(lines)


def _tune_delta(tune: dict | None) -> str:
    if not tune:
        return (
            "Пока сетки нет, рабочая точка — базовая. Она лежит в зоне temperature 0.2–0.5, "
            "а объём описания задан образцом в промпте: модель этого размера не умеет отмерять символы, "
            "зато повторяет длину показанной карточки.\n\n"
            "На что смотрит сетка:\n\n"
            "- temperature 0.0 / 0.3 / 0.7 — стабильность формата;\n"
            "- top_p 0.9 / 1.0 — тот же формат в паре с temperature;\n"
            "- top_k 20 / 50 / без ограничения — насколько жёстко отсекается хвост;\n"
            "- repetition_penalty 1.0 / 1.1 / 1.2 — повторы внутри tags и pros против связности описания;\n"
            "- max_tokens 250 / 350 / 500 — обрывы JSON снизу и лишние токены сверху."
        )
    baseline = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_tokens": 350,
        "json_mode": "guided",
    }
    grouped: dict[str, list] = {}
    for row in tune["rows"]:
        grouped.setdefault(row["parameter"], []).append(row)
    lines = [
        "Что изменилось относительно базовой точки: доля валидных, длина описания, выходные токены.",
        "",
    ]
    for parameter, rows in grouped.items():
        base_value = baseline[parameter]
        base = next((row for row in rows if row["value"] == base_value), rows[0])
        chosen_value = tune["chosen"][parameter]
        chosen_row = next(
            (
                row
                for row in rows
                if row["value"] == chosen_value
                or (chosen_value is None and row["value"] == "unlimited")
            ),
            base,
        )
        chosen_label = "без ограничения" if chosen_row["value"] == "unlimited" else chosen_row["value"]
        lines.append(
            f"- {parameter}: база {base['value']} — валидных {base['valid_rate'] * 100:.1f}%, "
            f"описание {base['avg_description_len']:.0f} символов, "
            f"{base['avg_output_tokens']:.0f} выходных токенов; "
            f"выбрано {chosen_label} — валидных {chosen_row['valid_rate'] * 100:.1f}%, "
            f"описание {chosen_row['avg_description_len']:.0f}, "
            f"{chosen_row['avg_output_tokens']:.0f} токенов."
        )
    return "\n".join(lines)


def _sweep_table(sweep: dict | None) -> str:
    if not sweep:
        return (
            "Таблица concurrency появится после `python run_batch.py --sweep 1,4,16,32 --sweep-n 40`.\n\n"
            "| concurrency | throughput, карт/с | latency, с | доля валидных | оценка времени на 300, с |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| — | после прогона | после прогона | после прогона | после прогона |"
        )
    lines = [
        f"Замер на {sweep['n']} товарах benchmark.jsonl, те же параметры, что у финального прогона.",
        "",
        "| concurrency | throughput, карт/с | latency, с | доля валидных | оценка времени на 300, с |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sweep["rows"]:
        lines.append(
            f"| {row['concurrency']} | {row['throughput']:.3f} | {row['avg_latency']:.2f} | "
            f"{row['valid_rate'] * 100:.1f}% | {row['time_for_300_sec']} |"
        )
    lines.append("")
    lines.append(
        f"Рабочая точка — concurrency {sweep['chosen_concurrency']}. "
        "Беру наибольший уровень, который прибавляет хотя бы 8% пропускной способности "
        "к уже выбранному и не роняет долю валидных больше чем на 8 пунктов относительно concurrency 1. "
        "Дальше плато: лишняя параллельность почти не ускоряет пакет и только растит очередь."
    )
    return "\n".join(lines)


def _comparison_table(report: dict | None, api_full: dict | None) -> str:
    if not report or not report["run"].get("elapsed_sec"):
        return (
            "| | Локальная модель на vLLM | Модель через API |\n"
            "| --- | --- | --- |\n"
            "| Доля валидных карточек | после прогона | после прогона |\n"
            "| Токенов на карточку, вход/выход | после прогона | после прогона |\n"
            "| Время на карточку | после прогона | после прогона |\n"
            "| Стоимость 1000 карточек | после прогона | после прогона |"
        )
    run = report["run"]
    metrics = report["metrics"]
    api = report["api_comparison"]
    n = run["n"] or 1
    api_n = api["n"] or 1
    local_in = run["input_tokens"] / n
    local_out = run["output_tokens"] / n
    api_in = (api["input_tokens"] / api_n) if api["n"] else 0
    api_out = (api["output_tokens"] / api_n) if api["n"] else 0
    api_latency = (api_full or {}).get("avg_latency")
    if api_latency is None and api["n"] and api["elapsed_sec"]:
        api_latency = api["elapsed_sec"] / api["n"]
    api_name = api["model"] or "не запускалась"
    return "\n".join(
        [
            f"Локальный прогон — benchmark.jsonl, {n} товаров, concurrency {run['concurrency']}. "
            f"API — подвыборка dev.jsonl, {api['n']} товаров, модель `{api_name}`. "
            "Промпт один и тот же. В цену API подставлены токены из `usage` ответа OpenRouter "
            "и условный прайс 20 ₽ / 60 ₽ за миллион входных и выходных токенов. "
            "Токены локальной модели в эту формулу не подставляются: токенизация другая.",
            "",
            "| | Локальная модель на vLLM | Модель через API |",
            "| --- | --- | --- |",
            f"| Доля валидных карточек | {metrics['valid_rate'] * 100:.1f}% | {_pct(api['valid_rate']) if api['n'] else 'не запускалась'} |",
            f"| Токенов на карточку, вход/выход | {local_in:.0f} / {local_out:.0f} | "
            + (f"{api_in:.0f} / {api_out:.0f}" if api["n"] else "—")
            + " |",
            f"| Время на карточку | {run['avg_latency']:.2f} с | "
            + (f"{api_latency:.2f} с" if api["n"] and api_latency is not None else "—")
            + " |",
            f"| Стоимость 1000 карточек | {metrics['cost_vllm_per_1000']:.2f} ₽ | "
            + (f"{api['cost_per_1000']:.2f} ₽" if api["n"] else "—")
            + " |",
        ]
    )


def _api_errors(api_full: dict | None) -> str:
    if not api_full:
        return (
            "Клиент повторяет 429 с экспоненциальной паузой от 1 до 30 секунд, не больше пяти раз подряд. "
            "5xx, таймаут и обрыв соединения обрабатываются так же. "
            "400 на незнакомый параметр (top_k, repetition_penalty, seed, guided_json) снимает этот параметр "
            "и повторяет тот же запрос; если не принимается json-схема, режим спускается "
            "guided → json_object → обычный текст. "
            "400 из-за длины контекста ужимает бюджет отзывов вдвое и не тратит на это попытку формата. "
            "После исчерпания повторов карточка помечается невалидной, пакет из-за одного отказа не останавливается. "
            "Какие коды реально пришли, будет видно в `outputs/api_comparison.json` после этапа 6."
        )
    errors = api_full.get("http_errors") or {}
    fired = {code: count for code, count in errors.items() if count}
    if not fired:
        seen = "В прогоне API ни один запрос не завершился ошибкой HTTP: счётчики 429, 400 и 5xx нулевые."
    else:
        parts = ", ".join(f"{code}: {count}" for code, count in sorted(fired.items()))
        seen = f"За время прогона клиент поймал и обработал ответы: {parts}."
    mode = api_full.get("json_mode", "object")
    return (
        f"{seen} Итоговый режим JSON на API — `{mode}`. "
        "429, 5xx и таймаут повторяются с паузой 1, 2, 4, … секунд, не больше пяти раз. "
        "400 из-за параметра, которого нет у внешней модели, убирает top_k, repetition_penalty и seed "
        "и сразу повторяет запрос. Если схема guided_json не поддерживается, остаётся `json_object`, "
        "а если не поддерживается и он — обычный текст и разбор через json.loads. "
        "Слишком длинный для модели контекст ужимается и запрашивается снова. "
        "Карточка, которую так и не удалось получить, остаётся невалидной и не валит остальные."
    )


def _decision(report: dict | None) -> str:
    formula = (
        "Обе цены — из формул условия.\n\n"
        "Свой сервер, время и число карточек полного прогона benchmark.jsonl:\n\n"
        "`(время_прогона_сек / 3600 × 65) / число_карточек × 1000`\n\n"
        "API, токены и число карточек прогона этапа 6:\n\n"
        "`(входные_токены × 20 + выходные_токены × 60) / 1 000 000 / число_карточек × 1000`\n\n"
        "Плановый объём — 50 000 карточек в месяц. Порог приёмки — не дороже 25 ₽ за 1000 "
        "на выбранном способе. Аренда T4 считается и за простой: если ВМ поднимать под каждый "
        "короткий запуск, холодный старт оплачивается, даже когда карточек мало."
    )
    if not report:
        return (
            formula
            + "\n\nЧисла подставятся после бенчмарка и этапа 6. "
            "Пока их нет, решение такое: считать маржинальную цену своего сервера по формуле выше "
            "и сравнивать её с ценой API. При плановых 50 000 карточках сервер, который включают "
            "на время пакета и выключают, почти наверняка дешевле помесячной аренды без выключения "
            "и дешевле API, если карточка на GPU выходит заметно дешевле 25 ₽."
        )
    if not report["api_comparison"].get("n"):
        run = report["run"]
        metrics = report["metrics"]
        per_vllm = metrics["cost_vllm_per_1000"]
        throughput = metrics["throughput"] or 0.0
        plan_vllm = monthly_costs(PLAN_VOLUME, per_vllm, 0.0)["vllm"]
        always_on_rub = GPU_RUB_PER_HOUR * HOURS_IN_MONTH
        parts = [
            formula,
            f"Полный прогон: {run['n']} карточек за {run['elapsed_sec']:.1f} с, "
            f"это {per_vllm:.2f} ₽ за 1000 на своём сервере. "
            f"На 50 000 карточек, если платить только за время генерации, это {plan_vllm:.0f} ₽.",
        ]
        if throughput:
            one_session = session_cost(PLAN_VOLUME, throughput, 1, round_up_hour=False)
            one_session_hour = session_cost(PLAN_VOLUME, throughput, 1, round_up_hour=True)
            daily = session_cost(PLAN_VOLUME, throughput, 30, round_up_hour=True)
            parts.append(
                f"При измеренной скорости {throughput:.3f} карт/с один месячный пакет на 50 000 карточек "
                f"занимает {PLAN_VOLUME / throughput / 3600:.1f} ч чистой генерации плюс старт: "
                f"{one_session:.0f} ₽ при посекундной оплате и {one_session_hour:.0f} ₽, "
                f"если незавершённый час округляется вверх. "
                f"Тридцать ежедневных включений с округлением до часа стоят {daily:.0f} ₽, "
                f"потому что почти каждый день оплачивается полный час простоя после короткого пакета."
            )
        parts.append(
            "Прогон через OpenRouter не выполнен: ключ OPENROUTER_API_KEY в окружении не задан. "
            "Объём, ниже которого API дешевле из-за простоя GPU, по этим данным не считается: "
            "в формулу API входят токены usage ответа OpenRouter, а не токены Qwen. "
            f"Порог 25 ₽ на своём сервере выполнен ({per_vllm:.2f} ₽ за 1000). "
            f"Постоянно включённая T4 стоит {always_on_rub:.0f} ₽ в месяц независимо от числа карточек, "
            "поэтому на плановые 50 000 карточек сервер включают на пакет и выключают."
        )
        return "\n\n".join(parts)
    run = report["run"]
    metrics = report["metrics"]
    api = report["api_comparison"]
    per_vllm = metrics["cost_vllm_per_1000"]
    per_api = api["cost_per_1000"]
    plan = monthly_costs(PLAN_VOLUME, per_vllm, per_api)
    startup = break_even_with_startup(per_vllm, per_api)
    always_on = break_even_always_on(per_api)
    always_on_rub = GPU_RUB_PER_HOUR * HOURS_IN_MONTH
    throughput = metrics["throughput"] or 0.0
    parts = [
        formula,
        "",
        f"Полный прогон: {run['n']} карточек за {run['elapsed_sec']:.1f} с, "
        f"это {per_vllm:.2f} ₽ за 1000 на своём сервере.",
        f"API `{api['model']}` на {api['n']} карточках потратил "
        f"{api['input_tokens']} входных и {api['output_tokens']} выходных токенов, "
        f"это {per_api:.2f} ₽ за 1000.",
        f"На 50 000 карточек, если платить только за время генерации: "
        f"свой сервер {plan['vllm']:.0f} ₽/мес, API {plan['api']:.0f} ₽/мес.",
    ]
    if startup is None:
        parts.append(
            "Карточка на API дешевле карточки на GPU, поэтому внешний API выгоднее на любом объёме, "
            "пока сервер не станет дешевле по токенам или по времени."
        )
    else:
        parts.append(
            f"Если ВМ поднимают под один прогон и закладывают 3 минуты холодного старта, "
            f"API дешевле только пока в месяц меньше {startup:.0f} карточек. "
            f"Выше этой отметки разница в цене карточки перекрывает разовый старт."
        )
    if always_on is not None:
        parts.append(
            f"Если ту же T4 держать включённой весь месяц, аренда стоит {always_on_rub:.0f} ₽ "
            f"независимо от числа карточек. API остаётся дешевле, пока объём ниже {always_on:.0f} карточек в месяц. "
            f"50 000 до этой границы не дотягивает, поэтому постоянно включённый GPU на плановом объёме дороже API."
        )
    if throughput:
        one_session = session_cost(PLAN_VOLUME, throughput, 1, round_up_hour=False)
        one_session_hour = session_cost(PLAN_VOLUME, throughput, 1, round_up_hour=True)
        daily = session_cost(PLAN_VOLUME, throughput, 30, round_up_hour=True)
        parts.append(
            f"При измеренной скорости {throughput:.3f} карт/с один месячный пакет на 50 000 карточек "
            f"занимает {PLAN_VOLUME / throughput / 3600:.1f} ч чистой генерации плюс старт: "
            f"{one_session:.0f} ₽ при посекундной оплате и {one_session_hour:.0f} ₽, "
            f"если незавершённый час округляется вверх. "
            f"Тридцать ежедневных включений с округлением до часа стоят {daily:.0f} ₽, "
            f"потому что почти каждый день оплачивается полный час простоя после короткого пакета."
        )
    chosen_cost = per_vllm if per_vllm <= per_api else per_api
    if chosen_cost <= 25 and per_vllm <= per_api:
        recommendation = (
            f"Итог для 50 000 карточек в месяц: свой сервер. "
            f"{per_vllm:.2f} ₽ за 1000 ниже и порога 25 ₽, и цены API {per_api:.2f} ₽. "
            "ВМ включают на пакет и выключают. Постоянно держать T4 не нужно: "
            "на этом объёме простоя больше, чем работы. API оставляем для объёма, "
            "при котором отдельный запуск GPU не окупает даже холодный старт."
        )
    elif chosen_cost <= 25:
        recommendation = (
            f"Итог: на этой задаче дешевле API, {per_api:.2f} ₽ за 1000 против "
            f"{per_vllm:.2f} ₽ на своём сервере. Порог 25 ₽ выполняется."
        )
    else:
        recommendation = (
            f"Итог: выбранный вариант выходит {chosen_cost:.2f} ₽ за 1000, это выше порога 25 ₽. "
            "Нужно поднимать пропускную способность или снижать число повторов."
        )
    parts.append(recommendation)
    return "\n\n".join(parts)


def _measurements(report: dict | None, diagnostics: dict | None) -> str:
    if not report:
        return (
            "- доля валидных карточек: после прогона benchmark.jsonl, разбивка брака по причинам там же\n"
            "- скорость: таблица в разделе замеров concurrency, затем полный прогон в выбранной точке\n"
            "- время на весь бенчмарк: после прогона, порог — не дольше 8 минут\n"
            "- стоимость: свой сервер по времени полного прогона, API по токенам этапа 6, порог — 25 ₽ за 1000"
        )
    metrics = report["metrics"]
    run = report["run"]
    reasons = metrics["errors_by_reason"] or {}
    reason_text = ", ".join(f"{name}: {count}" for name, count in reasons.items()) or "нарушений нет"
    long = ""
    if diagnostics:
        long = (
            f" Длинный вход: максимум {diagnostics.get('max_first_prompt_tokens')} токенов "
            f"на карточку, свыше 1500 — {diagnostics.get('n_over_1500')}, "
            f"с отброшенными отзывами — {diagnostics.get('n_truncated')}."
        )
    return "\n".join(
        [
            f"- доля валидных карточек: {metrics['valid']} из {run['n']} "
            f"({metrics['valid_rate'] * 100:.1f}%) на benchmark.jsonl, concurrency {run['concurrency']}. "
            f"Брак по причинам: {reason_text}.",
            f"- скорость: рабочая точка concurrency {run['concurrency']}, "
            f"throughput {metrics['throughput']:.3f} карт/с, средняя latency {run['avg_latency']:.2f} с. "
            "Таблица throughput и latency по уровням concurrency приведена в этом разделе.",
            f"- время на весь бенчмарк: {metrics['time_for_300_sec']} с "
            f"({metrics['time_for_300_sec'] / 60:.1f} мин) при n = {run['n']}.",
            f"- стоимость: на своём сервере {metrics['cost_vllm_per_1000']:.2f} ₽/1000 "
            f"из полного прогона ({run['elapsed_sec']} с, {run['n']} карточек); "
            + (
                f"через API {report['api_comparison']['cost_per_1000']:.2f} ₽/1000 "
                f"из прогона этапа 6 ({report['api_comparison']['n']} карточек, "
                f"{report['api_comparison']['input_tokens']} входных и "
                f"{report['api_comparison']['output_tokens']} выходных токенов)."
                if report["api_comparison"].get("n")
                else "сравнение с API не запускалось: в окружении не было OPENROUTER_API_KEY."
            ),
            f"- среднее число запросов на карточку, включая повтор при браке формата: {run['avg_attempts']}.{long}",
        ]
    )


def render_readme(report: dict | None = None) -> str:
    if report is None:
        stored = _load("report.json")
        report = stored
    tune = _load("tune.json")
    sweep = _load("sweep.json")
    diagnostics = _load("diagnostics.json")
    api_full = _load("api_comparison.json")
    chosen = (tune or {}).get("chosen") or {}
    temperature = chosen.get("temperature", 0.3)
    top_p = chosen.get("top_p", 0.9)
    top_k = chosen.get("top_k", 50)
    top_k_text = "без ограничения" if top_k is None else str(top_k)
    repetition = chosen.get("repetition_penalty", 1.1)
    max_tokens = chosen.get("max_tokens", 350)
    attempts = chosen.get("max_attempts", 2)
    json_mode = chosen.get("json_mode", "guided")
    attempts_why = (
        f"До {attempts} запросов на карточку: первая попытка и "
        + ("ещё одна коррекция" if attempts == 2 else f"до {attempts - 1} коррекций")
        + ". В повтор уходит тот же диалог и список конкретных нарушений "
        "(нет поля, неверная длина, не то число предложений или пунктов). "
        "Дальше повторять невыгодно: каждая лишняя попытка на хвосте брака добавляет и время, и токены, "
        "а порог 80% уже должен держаться за счёт промпта и одной правки. "
        "Если проверка на всём dev оказывается ниже 80%, скрипт подбора сам поднимает лимит до 3 "
        "и только потом фиксирует конфиг."
    )
    return f"""# Отчёт по проекту

Сервис собирает карточку товара для витрины «Полки» из названия, категории, характеристик и отзывов. Карточка валидна, когда витрина может подставить её в шаблон: один JSON-объект, пять полей, `product_id` того же товара, описание из 2–5 предложений и 120–700 символов, списки pros, cons и tags нужной длины. Смысл текста в метрику не входит.

Модель — Qwen2.5-1.5B-Instruct на vLLM. Внешнее сравнение — instruct-модель через OpenRouter, по умолчанию `openai/gpt-oss-20b`, ключ только из `OPENROUTER_API_KEY`.

## Финальный конфиг

- Модель `Qwen/Qwen2.5-1.5B-Instruct`, dtype `half` (float16), контекст 4096, `--gpu-memory-utilization 0.9`, `--enforce-eager`, порт 8000. Параметры генерации: temperature {temperature}, top_p {top_p}, top_k {top_k_text}, repetition_penalty {repetition}, max_tokens {max_tokens}. JSON-режим: `{json_mode}`. `top_k` и `repetition_penalty` уходят в vLLM через `extra_body`.
- Стратегия длинного входа: снять шаблонные приставки отзывов («Коротко:», «Пользуюсь месяц.», «В целом» и соседние), схлопнуть одинаковые формулировки и оставить частоту. В бюджет 1500 токенов, посчитанный токенайзером модели вместе с шаблоном чата, на этих данных входят все уникальные мнения. Отдельный запрос на сжатие не используется.
- Повторов при браке формата: {attempts}. {attempts_why}

Запуск сервера:

```bash
{VLLM_COMMAND}
```

Готовность: `curl -s http://localhost:8000/v1/models`.

## Параметры генерации

{_tune_table(tune)}

{_tune_delta(tune)}

## Сравнение с моделью через API

{_comparison_table(report, api_full)}

Ошибки API и обработка:

{_api_errors(api_full)}

В latency API входит сеть и очередь провайдера, у локальной модели этого слагаемого нет. Сравнивать скорость по этому столбцу как чистую скорость модели нельзя; для решения важны цена и доля валидных. Качество на задаче видно по доле валидных на одинаковом промпте. Что дешевле — по столбцу стоимости 1000 карточек и по расчёту в последнем разделе.

## Замеры

Скорость по concurrency:

{_sweep_table(sweep)}

Итог полного прогона:

{_measurements(report, diagnostics)}

Пороги приёмки: доля валидных не ниже 80%, 300 карточек не дольше 8 минут, выбранный способ не дороже 25 ₽ за 1000.

## Длинный вход

У большинства товаров 4–25 отзывов. У популярных в benchmark их 90–220, суммарно до ~12 тысяч символов. Это больше 1500 токенов, если класть ленту как есть. Отзывы при этом сильно повторяются: одна и та же оценка приходит с приставками «Коротко:», «Что могу сказать:», «Пользуюсь месяц.», «Купил на замену старому.», «Брал для себя.», «В целом».

Стратегия — очистка и дедупликация, без второго запроса к модели.

1. Приставки снимаются, пробелы схлопываются, одинаковые формулировки считаются одним мнением. Рядом сохраняется частота, чтобы модель видела, что сказали 20 покупателей, а что трое.
2. Если промпт с частотами не влезает в 1500 токенов, частоты убираются.
3. Если не влезает и так, бинарным поиском оставляется смесь: около 70% слотов — самые частые мнения, остальные — самые редкие. Редкие нужны, потому что минусы в этих отзывах встречаются реже плюсов, и обрезка «только топ» оставила бы блок минусов без опоры.
4. Токены считаются `apply_chat_template` токенайзера Qwen, то есть так же, как их увидит сервер.

Отдельное сжатие отзывов моделью на длинных товарах удвоило бы число запросов примерно у 15% каталога и добавило бы и время, и выходные токены. После дедупликации терять уже почти нечего: остаются сами оценки, а не их копии.

{_budget_paragraph()}

## Оптимизации

Точность — float16, флаг `--dtype half` из условия запуска.

Qwen2.5-1.5B в float16 занимает около 3 ГБ весов. На T4 с 16 ГБ это не упирается в память даже при контексте 4096 и запасе под батч, поэтому INT4 не покупает место под более крупную модель. Узкое место приёмки — доля валидного JSON, а не пропускная способность: порог 8 минут на 300 карточек для модели такого размера достижим и в float16. Квантизация в INT4 может ускорить чтение весов, но на 1.5B чаще растит долю обрывов и кривых списков. Каждый такой брак — это повторный запрос, и он съедает выигрыш по скорости. Пока полный прогон укладывается в 8 минут и в 25 ₽ за 1000, оставляю float16. Отдельный прогон INT4 имеет смысл только если float16 упрётся в один из этих двух порогов.

## Стоимость и решение

{_decision(report)}

## Как воспроизвести

```bash
# сервер, в отдельном терминале
{VLLM_COMMAND}

# данные, сетка, бенчмарк, API, отчёт
export OPENROUTER_API_KEY=ваш_ключ   # только в окружении, не в коде
bash run_on_vm.sh
```

По отдельности: `python tune_params.py`, `python run_batch.py --sweep 1,4,16,32 --sweep-n 40`, `python run_batch.py`, `python compare_api.py`, `python check_metrics.py --write-readme`.

Проверка формул и валидатора без GPU: `python check_metrics.py --self-check` и `python -m unittest tests.test_core`.

Артефакты лежат в `outputs/`: `predictions.jsonl` — карточка на строку, `report.json` — факты прогона, метрики и сравнение с API. Метрики в `report.json` пересчитываются из времени, токенов и повторной валидации карточек, а не записываются вручную.
"""
