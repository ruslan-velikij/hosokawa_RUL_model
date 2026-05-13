## Быстрый запуск основного сценария

```bash
python hosokawa_RUL_models.py
````

Далее в меню:

```text
1) Распаковать архив — при необходимости
2) Убрать дубли — при необходимости
3) Обработать журналы ТОИР — при необходимости
4) Выбрать эксперимент:
   3 — основной результат для мельницы
   7 — дополнительный результат для компактора с SSD
```

# Hosokawa RUL / прогнозирование технических событий оборудования

Проект предназначен для подготовки данных и построения моделей машинного обучения для прогнозирования приближения технических событий оборудования Hosokawa по временным рядам технологических сигналов.

В работе рассматриваются два основных узла оборудования:

- `mill` — мельница;
- `compactor` — компактор, шнек, валки, зона прессования/грануляции.

Целевая задача формулируется как бинарная классификация: определить, наступит ли техническое событие по выбранному узлу в ближайшие 24, 48 или 72 часа. Основной рабочий горизонт прогнозирования — `event_in_72h`.

## Общая логика проекта

Проект объединяет два типа данных:

1. Временные ряды оборудования из parquet-файлов.
2. Журналы ТОИР из Excel-файлов.

На основе журналов ТОИР формируются события по узлам оборудования. Затем эти события связываются с временными рядами по времени и номеру установки `N_Hosokawa`. После этого строятся целевые признаки:

- `event_in_24h` — событие наступит в ближайшие 24 часа;
- `event_in_48h` — событие наступит в ближайшие 48 часов;
- `event_in_72h` — событие наступит в ближайшие 72 часа;
- `time_to_next_event_hours` — время до ближайшего события в часах;
- `pre_event_window` — предсобытийное окно, совпадает с `event_in_72h`.

Далее данные очищаются, разбиваются на train/test по времени, дополняются производными признаками и используются для обучения моделей классификации.

## Структура проекта

```text
hosokawa_RUL_model/
├──_diagnostics/
    ├── 1_rows_and_columns.py
    ├── 2_1_base_analisys.py
    ├── 2_2_check_dt.py
    ├── 2_3_check_duplicates.py
    ├── 2_5_check_N_Hosokawa.py
    └── check_valkispeed_granulatorspeed.py
├── hosokawa_RUL_models.py
├── models/
│   ├── 2_4_remove_full_duplicates.py
│   ├── 3_1_merge_jtiny.py
│   ├── 3_2_clean_jtiny_merged.py
│   ├── 3_3_filter_hosokawa_events.py
│   ├── 3_4_classify_hosokawa_nodes.py
│   ├── 3_5_quality_label_events.py
│   ├── 4_1_link_events_to_timeseries.py
│   ├── 4_1_link_events_to_timeseries_strict.py
│   ├── 4_2_check_labeling.py
│   ├── 4_3_prepare_labeled_datasets.py
│   ├── 4_3_prepare_labeled_datasets_rolling.py
│   ├── 4_4_train_baseline_event_classifier.py
│   ├── 4_4_train_baseline_event_classifier_real.py
│   └── 4_5_tune_best_model.py
├── processed/
├── toir_hosokawa/
├── results/
└── requirements.txt
```

Назначение папок:

* `models/` — Python-скрипты обработки данных, разметки и обучения моделей;
* `processed/` — исходные parquet-файлы, очищенные parquet-файлы и промежуточные подготовленные датасеты;
* `toir_hosokawa/` — исходные Excel-журналы ТОИР и производные Excel-файлы после обработки;
* `results/` — результаты экспериментов: метрики, важность признаков, сводные JSON/CSV-файлы;
* `hosokawa_RUL_models.py` — управляющий консольный скрипт для запуска основных этапов и экспериментов;
* `_diagnostics/` — содержит разовые проверочные скрипты для первичного анализа данных. Они не используются в основной цепочке обучения, но оставлены для воспроизводимости технической разведки parquet-файлов.

## Входные данные

В проекте используются следующие основные parquet-файлы:

```text
processed/all_data_2020_2025.parquet
processed/all_data_2020_2025_with_ssd.parquet
```

Также присутствуют дополнительные parquet-файлы, но не используются:

```text
processed/data_2020_ssd_paneli_merged.parquet
processed/data_paneli_2020.parquet
processed/data_ssd_2020.parquet
processed/data_2024_ssd.parquet
processed/data_2025_am.parquet
processed/data_2025_ssd.parquet
```

Основной датасет:

```text
all_data_2020_2025.parquet
```

Расширенный SSD-датасет:

```text
all_data_2020_2025_with_ssd.parquet
```

SSD-датасет дополнительно используется для проверки влияния признаков:

* `ValkiSpeed`;
* `GranulatorSpeed`.

## Установка зависимостей

### Linux

Создать и активировать виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

### Windows

Создать и активировать виртуальное окружение:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Установить зависимости:

```powershell
pip install -r requirements.txt
```

## Управляющий запуск

Основной запуск выполняется из корня проекта:

```bash
python hosokawa_RUL_models.py
```

Скрипт последовательно предлагает:

1. Распаковать архив `MSZ.zip`.
2. Удалить полные дубли из parquet-файлов.
3. Обработать журналы ТОИР.
4. Выбрать эксперимент для запуска.

## Этап 1. Распаковка данных

Управляющий скрипт распаковывается архив `MSZ.zip`.

После распаковки:

* содержимое `processed.7z` переносится в папку `processed/`;
* содержимое `toir_hosokawa.7z` переносится в папку `toir_hosokawa/`.

Этот шаг можно пропустить и подготовить папки вручную.

## Этап 2. Удаление полных дублей

Удаление дублей выполняется скриптом:

```bash
python models/2_4_remove_full_duplicates.py --input processed/all_data_2020_2025.parquet
python models/2_4_remove_full_duplicates.py --input processed/all_data_2020_2025_with_ssd.parquet
```

На выходе формируются файлы:

```text
processed/all_data_2020_2025_no_duplicates.parquet
processed/all_data_2020_2025_with_ssd_no_duplicates.parquet
```

Полные дубли удаляются по всей строке.

## Этап 3. Обработка журналов ТОИР

Обработка журналов выполняется цепочкой:

```bash
python models/3_1_merge_jtiny.py
python models/3_2_clean_jtiny_merged.py
python models/3_3_filter_hosokawa_events.py
python models/3_4_classify_hosokawa_nodes.py
python models/3_5_quality_label_events.py
```

Назначение файлов:

| Скрипт                           | Назначение                                                                |
| -------------------------------- | ------------------------------------------------------------------------- |
| `3_1_merge_jtiny.py`             | объединяет листы `jtiny` из нескольких Excel-журналов                     |
| `3_2_clean_jtiny_merged.py`      | очищает объединённый журнал, приводит даты, оставляет события с 2020 года |
| `3_3_filter_hosokawa_events.py`  | отбирает события Hosokawa и присваивает класс события                     |
| `3_4_classify_hosokawa_nodes.py` | распределяет события по узлам `mill`, `compactor`, `other`                |
| `3_5_quality_label_events.py`    | формирует строгую разметку событий для отдельного эксперимента            |

Основной файл событий после обработки:

```text
toir_hosokawa/jtiny_hosokawa_events_by_node.xlsx
```

Файл строгой разметки:

```text
toir_hosokawa/jtiny_hosokawa_events_quality_labeled.xlsx
```

## Этап 4. Увязка событий с временными рядами

Основная увязка событий выполняется скриптом:

```bash
python models/4_1_link_events_to_timeseries.py --input processed/all_data_2020_2025_no_duplicates.parquet
```

Для SSD-датасета:

```bash
python models/4_1_link_events_to_timeseries.py --input processed/all_data_2020_2025_with_ssd_no_duplicates.parquet
```

На выходе формируются:

```text
processed/compactor_dataset_labeled.parquet
processed/mill_dataset_labeled.parquet
```

Эти файлы содержат временные ряды по каждому узлу и целевые признаки `event_in_24h`, `event_in_48h`, `event_in_72h`.

Для строгой разметки используется отдельный скрипт:

```bash
python models/4_1_link_events_to_timeseries_strict.py --input processed/all_data_2020_2025_no_duplicates.parquet
```

## Этап 5. Проверка разметки

Проверка количества положительных меток:

```bash
python models/4_2_check_labeling.py
```

Скрипт выводит:

* размер датасетов;
* количество строк с `event_in_24h`;
* количество строк с `event_in_48h`;
* количество строк с `event_in_72h`;
* описание `time_to_next_event_hours`.

## Этап 6. Подготовка данных к обучению

Основная подготовка данных:

```bash
python models/4_3_prepare_labeled_datasets.py
```

Что выполняется:

* проверка `DT`, `N_Hosokawa` и целевых признаков;
* разбиение на train/test по времени;
* удаление почти пустых и константных признаков;
* добавление календарных признаков;
* добавление лагов и разностей `lag1`, `diff1`;
* добавление индикаторов пропусков;
* заполнение пропусков;
* сохранение train/test-файлов.

Граница разбиения:

```text
train: DT < 2025-01-01
test:  DT >= 2025-01-01
```

На выходе:

```text
processed/compactor_train_prepared.parquet
processed/compactor_test_prepared.parquet
processed/mill_train_prepared.parquet
processed/mill_test_prepared.parquet
```

Версия с оконными признаками:

```bash
python models/4_3_prepare_labeled_datasets_rolling.py
```

Она дополнительно формирует оконные признаки по технологическим сигналам:

* среднее значение в окне;
* стандартное отклонение;
* диапазон значений.

## Этап 7. Обучение моделей

Основная модель с реальными технологическими признаками:

```bash
python models/4_4_train_baseline_event_classifier_real.py
```

В этой версии из обучения исключаются:

* `N_Hosokawa`;
* `Regim`;
* календарные признаки;
* служебные признаки времени;
* признаки факта пропуска `*_was_missing`;
* целевые признаки.

В модели остаются реальные технологические сигналы оборудования и их производные признаки.

Для раннего эксперимента со всеми признаками используется:

```bash
python models/4_4_train_baseline_event_classifier.py
```

Основная модель:

```text
HistGradientBoostingClassifier
```

Дополнительно обучается базовая модель:

```text
DummyClassifier
```

Она нужна для сравнения с наивным уровнем.

## Метрики качества

В проекте используются следующие метрики:

| Метрика                    | Назначение                                                             |
| -------------------------- | ---------------------------------------------------------------------- |
| `ROC-AUC`                  | оценивает способность модели отделять предсобытийные строки от обычных |
| `Average Precision` / `AP` | основная метрика для редкого положительного класса                     |
| `Precision`                | доля верных предупреждений среди всех предупреждений модели            |
| `Recall`                   | доля найденных предсобытийных состояний                                |
| `F1`                       | баланс между precision и recall                                        |
| `AP lift`                  | отношение AP модели к базовой доле положительного класса               |

Для задачи прогнозирования редких технических событий особенно важна `Average Precision`, так как положительный класс занимает небольшую долю данных.

## Эксперименты

Управляющий скрипт позволяет запустить 7 сценариев.

### Эксперимент 1. Без SSD, все события

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: все события из журналов ТОИР
Модель: baseline со всеми признаками
```

Цепочка:

```bash
python models/4_1_link_events_to_timeseries.py --input processed/all_data_2020_2025_no_duplicates.parquet
python models/4_2_check_labeling.py
python models/4_3_prepare_labeled_datasets.py
python models/4_4_train_baseline_event_classifier.py
```

Результаты:

```text
results/01_no_ssd_all_events
```

### Эксперимент 2. Только события «Неисправность/отказ»

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: только Неисправность/отказ
Модель: baseline со всеми признаками
```

Результаты:

```text
results/02_no_ssd_failures_all_features
```

### Эксперимент 3. Неисправности/отказы + реальные признаки

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: только Неисправность/отказ
Признаки: реальные технологические признаки
Модель: HistGradientBoostingClassifier
```

Результаты:

```text
results/03_no_ssd_failures_real_features
```

Этот эксперимент является основным для узла мельницы.

### Эксперимент 4. Неисправности/отказы + реальные признаки + оконные признаки

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: только Неисправность/отказ
Признаки: реальные технологические признаки + оконные признаки
```

Результаты:

```text
results/04_no_ssd_failures_real_features_rolling
```

Эксперимент проверяет, улучшается ли качество модели при добавлении динамических признаков за скользящие окна.

### Эксперимент 5. Строгая разметка событий

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: только strict_failure с label_quality = good
Признаки: реальные технологические признаки
```

Результаты:

```text
results/05_strict_labeling_real_features
```

Эксперимент проверяет, улучшает ли качество модели более жёсткая фильтрация событий ТОИР.

### Эксперимент 6. Строгая разметка + подбор гиперпараметров

```text
Датасет: all_data_2020_2025_no_duplicates.parquet
События: строгая разметка
Дополнительно: подбор гиперпараметров для модели мельницы
```

Результаты:

```text
results/06_strict_labeling_hyperparameter_tuning
```

Подбор гиперпараметров выполняется скриптом:

```bash
python models/4_5_tune_best_model.py
```

### Эксперимент 7. SSD-датасет + неисправности/отказы + реальные признаки

```text
Датасет: all_data_2020_2025_with_ssd_no_duplicates.parquet
События: только Неисправность/отказ
Признаки: реальные технологические признаки
Дополнительные признаки для compactor: ValkiSpeed, GranulatorSpeed
```

Результаты:

```text
results/07_ssd_failures_real_features
```

Этот эксперимент является дополнительным полезным результатом для узла компактора.

## Очистка промежуточных файлов

Перед запуском очередного эксперимента управляющий скрипт очищает промежуточные файлы в `processed/`, чтобы результаты разных сценариев не смешивались.

При этом сохраняются исходные и основные подготовленные parquet-файлы:

```text
all_data_2020_2025.parquet
all_data_2020_2025_no_duplicates.parquet
all_data_2020_2025_with_ssd.parquet
all_data_2020_2025_with_ssd_no_duplicates.parquet
data_*.parquet
```

Удаляются временные файлы вида:

```text
compactor_dataset_labeled.parquet
mill_dataset_labeled.parquet
compactor_train_prepared.parquet
mill_train_prepared.parquet
*_prepare_report.json
baseline_event_classifier/
baseline_event_classifier_real_signals_only/
tuned_best_model_mill_event_in_72h/
```

Итоговые результаты экспериментов сохраняются в `results/` и не удаляются.

## Итоговые выбранные варианты

По результатам экспериментов были выделены два наиболее полезных варианта.

### Основной результат для мельницы

```text
Узел: mill
Датасет: all_data_2020_2025_no_duplicates.parquet
События: Неисправность/отказ
Признаки: реальные технологические признаки
Модель: HistGradientBoostingClassifier
```

Этот вариант показал наиболее содержательный результат для узла мельницы.

### Дополнительный результат для компактора

```text
Узел: compactor
Датасет: all_data_2020_2025_with_ssd_no_duplicates.parquet
События: Неисправность/отказ
Признаки: реальные технологические признаки + SSD-признаки ValkiSpeed и GranulatorSpeed
Модель: HistGradientBoostingClassifier
```

Для компактора расширение признакового пространства за счёт SSD-данных дало более полезный результат, чем базовый датасет.

## Где смотреть результаты

В каждой папке эксперимента в `results/` сохраняются:

```text
summary_event_in_72h.json
compactor_event_in_72h_metrics.json
mill_event_in_72h_metrics.json
*_threshold_metrics.csv
*_feature_importance.csv
run_commands.txt
```

Файл `summary_event_in_72h.json` содержит сводку по моделям для обоих узлов.

Файлы `*_metrics.json` содержат подробные метрики по каждому узлу.

Файлы `*_threshold_metrics.csv` содержат precision, recall и F1 при разных порогах классификации.

Файлы `*_feature_importance.csv` содержат оценку важности признаков.

Файл `run_commands.txt` фиксирует команды, выполненные при запуске эксперимента.

## Краткий вывод

```text
В рамках проекта построен полный конвейер подготовки данных и обучения моделей для прогнозирования приближения технических событий оборудования Hosokawa. Лучший основной результат получен для узла мельницы на базовом датасете при использовании событий класса «Неисправность/отказ» и реальных технологических признаков. Для компактора более полезным оказался SSD-датасет, так как дополнительные признаки `ValkiSpeed` и `GranulatorSpeed` улучшают описание работы зоны валков и грануляции.
````

