# Поиск услуг: отбор кандидатов

Решение возвращает до 50 уникальных объявлений для поискового запроса.
В `solution.ipynb` сравниваются BM25, веса полей, word/character TF-IDF,
zero-shot dense и их объединение через RRF. Обучение моделей не используется.

## Результат

Выбран **BM25 + multilingual-e5-small → RRF**: по 200 кандидатов от каждого
канала, `c=60`, равные веса. Итоговая выдача — top-50, равенства разрешаются по `item_id`.

| Метод | Macro Recall@50 |
| --- | ---: |
| BM25 | 1,9168% |
| Word TF-IDF | 1,5504% |
| Character TF-IDF | 1,3516% |
| Dense | 1,8395% |
| **BM25 + dense, RRF** | **2,1566%** |

Прирост к BM25 — **+0,2399 п.п.**; скорректированный bootstrap-интервал
98,33%: **[+0,1580; +0,3237] п.п.** Все варианты сохранены в `experiments.csv`.

В validation отложены 20% уникальных нормализованных текстов, seed 42;
контексты одного текста остаются вместе. Позитивы вне корпуса учитываются
в знаменателе, поэтому потолок Recall@50 — около 6,13%. По доступным позитивам
итоговый Recall@50 составляет 33,60%. Конфигурация выбрана на этом же validation;
контекстные признаки не используются, качество на итоговых запросах неизвестно.

## Файлы

- `solution.ipynb` — исследование, проверки и генерация ответа.
- `split.csv`, `experiments.csv` — фиксированный split и результаты экспериментов.
- `requirements.txt` — версии зависимостей.

В `dataset/` нужны `benchmark_items.parquet` (189 212 объявлений),
`benchmark_queries.parquet` (2 452 запроса) и `train.parquet` (497 673 взаимодействия).
Каждое взаимодействие считается позитивом; исходные файлы не изменяются.

## Запуск

Python 3.12, команды из корня проекта:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name avito --display-name avito
```

Один раз загрузите веса:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    cache_dir=".cache/huggingface",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "sentencepiece*", "*.txt"],
    ignore_patterns=["onnx/*", "openvino/*"],
)
PY
```

Откройте `solution.ipynb`, выберите kernel `avito` и выполните все ячейки.
Индексы и embeddings пересоздаются; `answer.csv` обновляется автоматически.
После установки зависимостей и загрузки весов интернет не требуется.

`AVITO_DEVICE=auto` выбирает CUDA → MPS → CPU; доступны явные `cpu`, `cuda`, `mps`.
Переменную задавайте до запуска kernel. BM25 и TF-IDF всегда работают на CPU.
Сборку PyTorch выбирайте по [официальной инструкции](https://pytorch.org/get-started/locally/).
Полный прогон проверен на RTX 5090; CPU — отдельными проверками, реальный MPS не проверен.
Character TF-IDF требует нескольких ГиБ RAM; embeddings корпуса занимают 277 МиБ.
