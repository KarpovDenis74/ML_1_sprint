"""Готовая обёртка пакетного прогона из условия задания."""

import asyncio
import time


async def run_concurrent(items, concurrency, generate_one):
    """Отправляет запросы параллельно, не больше `concurrency` одновременно.

    generate_one — асинхронная функция, принимает один товар и возвращает
    словарь с результатом. Возвращает список результатов и общее время прогона.
    """
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(item):
        async with sem:
            return await generate_one(item)

    started = time.perf_counter()
    results = await asyncio.gather(*[wrapped(it) for it in items])
    return results, time.perf_counter() - started
