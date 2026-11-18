#!/usr/bin/env bash
# Полный прогон на ВМ: сетка параметров, выбор concurrency, бенчмарк, API, отчёт.
# Сервер vLLM нужно поднять отдельно, команда есть в README.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

mkdir -p data outputs
if [[ ! -f data/dev.jsonl || ! -f data/benchmark.jsonl ]]; then
  curl -fsSL -o data/dev.jsonl https://code.s3.yandex.net/ai-engineer/s1project/dev.jsonl
  curl -fsSL -o data/benchmark.jsonl https://code.s3.yandex.net/ai-engineer/s1project/benchmark.jsonl
fi

"$PYTHON" check_metrics.py --self-check
"$PYTHON" -m unittest tests.test_core -q

echo "Жду vLLM на http://localhost:8000 ..."
ready=0
for _ in $(seq 1 180); do
  if curl -sf http://localhost:8000/v1/models >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "vLLM не ответил. Запустите сервер командой из README и повторите." >&2
  exit 1
fi

"$PYTHON" tune_params.py
"$PYTHON" run_batch.py --sweep 1,4,16,32 --sweep-n 40
"$PYTHON" run_batch.py --input data/benchmark.jsonl

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  "$PYTHON" compare_api.py
else
  echo "OPENROUTER_API_KEY не задан: сравнение с API пропущено." >&2
  echo "Когда ключ будет в окружении, выполните: python compare_api.py && python check_metrics.py --write-readme" >&2
fi

"$PYTHON" check_metrics.py --write-readme
