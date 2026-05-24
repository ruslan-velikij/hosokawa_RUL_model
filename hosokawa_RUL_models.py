#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hosokawa_RUL_models.py

Консольный управляющий скрипт для проекта Hosokawa RUL.

Ожидаемая структура проекта:

hosokawa_RUL/
├── models/
│   ├── hosokawa_RUL_models.py
│   ├── 2_4_remove_full_duplicates.py
│   ├── 3_1_merge_jtiny.py
│   ├── ...
├── processed/
├── toir_hosokawa/
└── results/

Скрипт:
1. По желанию распаковывает основной архив MSZ.zip.
2. По желанию удаляет полные дубли из базового и SSD parquet.
3. По желанию запускает обработку журналов ТОИР.
4. По желанию запускает выбранные эксперименты 1–7.
5. По желанию запускает дополнительный RUL-анализ скриптами 5_1–5_4.
"""

from __future__ import annotations

import getpass
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()

if THIS_FILE.parent.name == "models":
    MODELS_DIR = THIS_FILE.parent
    PROJECT_ROOT = MODELS_DIR.parent
else:
    PROJECT_ROOT = THIS_FILE.parent
    MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR = PROJECT_ROOT / "processed"
TOIR_DIR = PROJECT_ROOT / "toir_hosokawa"
RESULTS_DIR = PROJECT_ROOT / "results"

BASIC_PARQUET = PROCESSED_DIR / "all_data_2020_2025.parquet"
BASIC_PARQUET_NODUP = PROCESSED_DIR / "all_data_2020_2025_no_duplicates.parquet"

SSD_PARQUET = PROCESSED_DIR / "all_data_2020_2025_with_ssd.parquet"
SSD_PARQUET_NODUP = PROCESSED_DIR / "all_data_2020_2025_with_ssd_no_duplicates.parquet"

RUN_LOG: list[str] = []


def print_header(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def ask_yes_no(question: str, default: str = "n") -> bool:
    default = default.lower().strip()
    suffix = " [y/N]: " if default == "n" else " [Y/n]: "

    while True:
        answer = input(question + suffix).strip().lower()
        if not answer:
            answer = default

        if answer in {"y", "yes", "д", "да"}:
            return True
        if answer in {"n", "no", "н", "нет"}:
            return False

        print("Введите y или n.")


def script_path(name: str) -> Path:
    path = MODELS_DIR / name
    if path.exists():
        return path

    root_path = PROJECT_ROOT / name
    if root_path.exists():
        return root_path

    raise FileNotFoundError(f"Не найден скрипт: {path}")


def project_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_python_script(name: str, args: list[str] | None = None) -> None:
    args = args or []
    path = script_path(name)

    cmd = [sys.executable, str(path), *args]
    RUN_LOG.append(" ".join(cmd))

    print()
    print("-" * 90)
    print("Запуск:")
    print(" ".join(cmd))
    print("-" * 90)

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def ensure_project_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOIR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_processed_base_file(path: Path) -> bool:
    """Возвращает True для исходных parquet и parquet после удаления дублей."""
    if not path.is_file():
        return False

    if path.suffix.lower() != ".parquet":
        return False

    name = path.name

    # Основные рабочие parquet: исходные и версии *_no_duplicates.
    if name.startswith("all_data_2020_2025"):
        return True

    # Дополнительные исходные parquet из архива.
    if name.startswith("data_"):
        return True

    return False


def clean_processed_intermediate_files() -> None:
    """Очищает processed от промежуточных файлов перед запуском эксперимента.

    Остаются только исходные parquet и parquet, полученные после удаления дублей.
    Удаляются labeled/prepared/train/test-файлы, отчеты подготовки и папки результатов
    обучения внутри processed. Итоговые результаты в папке results не трогаются.
    """
    print_header("Очистка processed перед экспериментом")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
    removed: list[str] = []

    for item in sorted(PROCESSED_DIR.iterdir(), key=lambda p: p.name):
        # Служебный файл для сохранения пустой папки processed в git.
        # Его нельзя удалять при очистке промежуточных результатов.
        if item.name == ".gitkeep":
            kept.append(item.name)
            continue

        if is_processed_base_file(item):
            kept.append(item.name)
            continue

        if item.is_dir():
            shutil.rmtree(item)
            removed.append(item.name + "/")
        else:
            item.unlink()
            removed.append(item.name)

    if kept:
        print("Оставлены базовые файлы:")
        for name in kept:
            print(f"  - {name}")
    else:
        print("Базовые parquet-файлы в processed не найдены.")

    if removed:
        print("\nУдалены промежуточные файлы и папки:")
        for name in removed:
            print(f"  - {name}")
    else:
        print("\nПромежуточных файлов для удаления не найдено.")


def begin_experiment(title: str) -> None:
    """Начинает эксперимент: сбрасывает лог команд и чистит processed."""
    RUN_LOG.clear()
    print_header(title)
    clean_processed_intermediate_files()


def copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Не найдена папка для копирования: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def copy_artifacts(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"ВНИМАНИЕ: папка результатов не найдена и не будет скопирована: {src}")
        return

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print(f"Результаты скопированы: {dst}")


def save_run_log(result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "run_commands.txt").write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")


def find_single_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"В распакованном архиве не найден файл {filename}")
    return matches[0]


def extract_7z(archive_path: Path, output_dir: Path, password: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = ["7z", "7zz", "7za"]
    last_error: Exception | None = None

    for command in candidates:
        try:
            subprocess.run(
                [
                    command,
                    "x",
                    str(archive_path),
                    f"-o{output_dir}",
                    f"-p{password}",
                    "-y",
                ],
                check=True,
            )
            return
        except FileNotFoundError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Не найден 7z/7zz/7za. Установи p7zip/7zip и повтори запуск."
    ) from last_error


def choose_extracted_source(extracted_root: Path, expected_dir_name: str) -> Path:
    direct = extracted_root / expected_dir_name
    if direct.exists() and direct.is_dir():
        return direct

    candidates = [p for p in extracted_root.rglob(expected_dir_name) if p.is_dir()]
    if candidates:
        return candidates[0]

    return extracted_root


def unpack_main_archive() -> None:
    print_header("1. Распаковка основного архива MSZ.zip")

    archive_input = input("Укажите путь к MSZ.zip: ").strip().strip('"')
    if not archive_input:
        raise SystemExit("Путь к архиву не указан.")

    archive_path = Path(archive_input).expanduser().resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"Архив не найден: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Ожидался zip-архив, получено: {archive_path}")

    password = getpass.getpass("Пароль от processed.7z и toir_hosokawa.7z: ")

    temp_dir = PROJECT_ROOT / "_tmp_unpack_MSZ"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    print(f"Распаковываю zip во временную папку: {temp_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(temp_dir)

    processed_7z = find_single_file(temp_dir, "processed.7z")
    toir_7z = find_single_file(temp_dir, "toir_hosokawa.7z")

    processed_extract = temp_dir / "_processed_extract"
    toir_extract = temp_dir / "_toir_extract"

    print(f"Распаковываю: {processed_7z.name}")
    extract_7z(processed_7z, processed_extract, password)

    print(f"Распаковываю: {toir_7z.name}")
    extract_7z(toir_7z, toir_extract, password)

    processed_source = choose_extracted_source(processed_extract, "processed")
    toir_source = choose_extracted_source(toir_extract, "toir_hosokawa")

    print(f"Переношу файлы в: {PROCESSED_DIR}")
    copy_dir_contents(processed_source, PROCESSED_DIR)

    print(f"Переношу файлы в: {TOIR_DIR}")
    copy_dir_contents(toir_source, TOIR_DIR)

    print("Распаковка завершена.")
    print(f"Временная папка оставлена для проверки: {temp_dir}")


def remove_duplicates_for_known_parquets() -> None:
    print_header("2. Удаление полных дублей из parquet")

    tasks = [BASIC_PARQUET, SSD_PARQUET]

    for input_path in tasks:
        if not input_path.exists():
            print(f"Пропуск: файл не найден: {input_path}")
            continue

        run_python_script(
            "2_4_remove_full_duplicates.py",
            ["--input", project_rel(input_path)],
        )


def process_toir_journals() -> None:
    print_header("3. Обработка журналов ТОИР")

    chain = [
        "3_1_merge_jtiny.py",
        "3_2_clean_jtiny_merged.py",
        "3_3_filter_hosokawa_events.py",
        "3_4_classify_hosokawa_nodes.py",
        "3_5_quality_label_events.py",
    ]

    for script in chain:
        run_python_script(script)


def set_event_class_filter(value: str) -> None:
    path = script_path("4_1_link_events_to_timeseries.py")
    text = path.read_text(encoding="utf-8")

    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(text, encoding="utf-8")

    if value == "all":
        new_line = 'EVENT_CLASS_FILTER: Optional[list[str]] = None'
    elif value == "failures":
        new_line = 'EVENT_CLASS_FILTER: Optional[list[str]] = ["Неисправность/отказ"]'
    else:
        raise ValueError(f"Неизвестный режим EVENT_CLASS_FILTER: {value}")

    pattern = r"(?m)^EVENT_CLASS_FILTER(?:\s*:\s*Optional\[list\[str\]\])?\s*=\s*.*$"
    new_text, count = re.subn(pattern, new_line, text, count=1)

    if count != 1:
        raise RuntimeError("Не удалось автоматически найти строку EVENT_CLASS_FILTER в 4_1_link_events_to_timeseries.py")

    path.write_text(new_text, encoding="utf-8")
    print(f"В файле {path.name} установлено: {new_line}")


def run_link_basic(input_parquet: Path) -> None:
    run_python_script(
        "4_1_link_events_to_timeseries.py",
        ["--input", project_rel(input_parquet)],
    )


def run_link_strict(input_parquet: Path) -> None:
    run_python_script(
        "4_1_link_events_to_timeseries_strict.py",
        ["--input", project_rel(input_parquet)],
    )


def run_check_prepare_train(
    prepare_script: str,
    train_script: str,
    result_source_dir_name: str,
    result_target_dir_name: str,
) -> None:
    run_python_script("4_2_check_labeling.py")
    run_python_script(prepare_script)
    run_python_script(train_script)

    target_dir = RESULTS_DIR / result_target_dir_name
    copy_artifacts(PROCESSED_DIR / result_source_dir_name, target_dir)
    save_run_log(target_dir)


def experiment_1_all_events() -> None:
    begin_experiment('Эксперимент 1: без SSD, все события')
    set_event_class_filter("all")
    run_link_basic(BASIC_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets.py",
        train_script="4_4_train_baseline_event_classifier.py",
        result_source_dir_name="baseline_event_classifier",
        result_target_dir_name="01_no_ssd_all_events",
    )


def experiment_2_failures_all_features() -> None:
    begin_experiment('Эксперимент 2: только события "Неисправность/отказ", все признаки')
    set_event_class_filter("failures")
    run_link_basic(BASIC_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets.py",
        train_script="4_4_train_baseline_event_classifier.py",
        result_source_dir_name="baseline_event_classifier",
        result_target_dir_name="02_no_ssd_failures_all_features",
    )


def experiment_3_failures_real_features() -> None:
    begin_experiment('Эксперимент 3: неисправности/отказы + реальные признаки')
    set_event_class_filter("failures")
    run_link_basic(BASIC_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets.py",
        train_script="4_4_train_baseline_event_classifier_real.py",
        result_source_dir_name="baseline_event_classifier_real_signals_only",
        result_target_dir_name="03_no_ssd_failures_real_features",
    )


def experiment_4_failures_real_features_rolling() -> None:
    begin_experiment('Эксперимент 4: неисправности/отказы + реальные признаки + оконные признаки')
    set_event_class_filter("failures")
    run_link_basic(BASIC_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets_rolling.py",
        train_script="4_4_train_baseline_event_classifier_real.py",
        result_source_dir_name="baseline_event_classifier_real_signals_only",
        result_target_dir_name="04_no_ssd_failures_real_features_rolling",
    )


def experiment_5_strict_labeling() -> None:
    begin_experiment('Эксперимент 5: строгая разметка событий')
    run_python_script("3_5_quality_label_events.py")
    run_link_strict(BASIC_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets.py",
        train_script="4_4_train_baseline_event_classifier_real.py",
        result_source_dir_name="baseline_event_classifier_real_signals_only",
        result_target_dir_name="05_strict_labeling_real_features",
    )


def experiment_6_strict_with_tuning() -> None:
    begin_experiment('Эксперимент 6: строгая разметка + подбор гиперпараметров')
    run_python_script("3_5_quality_label_events.py")
    run_link_strict(BASIC_PARQUET_NODUP)
    run_python_script("4_2_check_labeling.py")
    run_python_script("4_3_prepare_labeled_datasets.py")
    run_python_script("4_4_train_baseline_event_classifier_real.py")
    run_python_script("4_5_tune_best_model.py")

    target_dir = RESULTS_DIR / "06_strict_labeling_hyperparameter_tuning"
    target_dir.mkdir(parents=True, exist_ok=True)

    copy_artifacts(
        PROCESSED_DIR / "baseline_event_classifier_real_signals_only",
        target_dir / "baseline_event_classifier_real_signals_only",
    )
    copy_artifacts(
        PROCESSED_DIR / "tuned_best_model_mill_event_in_72h",
        target_dir / "tuned_best_model_mill_event_in_72h",
    )
    save_run_log(target_dir)


def experiment_7_ssd_failures_real_features() -> None:
    begin_experiment('Эксперимент 7: SSD-датасет + неисправности/отказы + реальные признаки')
    set_event_class_filter("failures")

    if not SSD_PARQUET_NODUP.exists():
        print(f"Файл {SSD_PARQUET_NODUP} не найден.")
        if SSD_PARQUET.exists() and ask_yes_no("Удалить дубли из SSD parquet сейчас?", default="y"):
            run_python_script(
                "2_4_remove_full_duplicates.py",
                ["--input", project_rel(SSD_PARQUET)],
            )
        else:
            raise FileNotFoundError(f"Не найден подготовленный SSD parquet: {SSD_PARQUET_NODUP}")

    run_link_basic(SSD_PARQUET_NODUP)
    run_check_prepare_train(
        prepare_script="4_3_prepare_labeled_datasets.py",
        train_script="4_4_train_baseline_event_classifier_real.py",
        result_source_dir_name="baseline_event_classifier_real_signals_only",
        result_target_dir_name="07_ssd_failures_real_features",
    )


EXPERIMENTS = {
    "1": {"name": 'без SSD, все события', "func": experiment_1_all_events},
    "2": {"name": 'только события "Неисправность/отказ"', "func": experiment_2_failures_all_features},
    "3": {"name": 'только события "Неисправность/отказ" + реальные признаки', "func": experiment_3_failures_real_features},
    "4": {"name": 'неисправности/отказы + реальные признаки + оконные признаки', "func": experiment_4_failures_real_features_rolling},
    "5": {"name": 'строгая разметка событий', "func": experiment_5_strict_labeling},
    "6": {"name": 'строгая разметка + подбор гиперпараметров', "func": experiment_6_strict_with_tuning},
    "7": {"name": 'SSD-датасет + неисправности/отказы + реальные признаки', "func": experiment_7_ssd_failures_real_features},
}


RUL_ANALYSIS_STEPS = [
    {
        "name": "Анализ интервалов между отказами",
        "script": "5_1_analyze_failure_intervals.py",
        "args": [],
    },
    {
        "name": "RUL-регрессия по календарному времени до отказа",
        "script": "5_2_train_rul_regressor.py",
        "args": [],
    },
    {
        "name": "Анализ ложных срабатываний классификационной модели",
        "script": "5_3_analyze_false_alarms.py",
        "args": [],
    },
    {
        "name": "Расчет рабочих часов до отказа",
        "script": "5_4_calculate_operating_hours_to_failure.py",
        "args": [],
    },
]


def choose_and_run_experiments() -> bool:
    print_header("4. Выбор эксперимента")

    print("Доступные сценарии:")
    for key, item in EXPERIMENTS.items():
        print(f"{key}. {item['name']}")

    print()
    print("Можно ввести один номер, несколько через запятую, all или n.")
    choice = input("Что запустить? ").strip().lower()

    if not choice or choice in {"n", "no", "нет"}:
        print("Запуск экспериментов пропущен.")
        return False

    if choice == "all":
        keys = list(EXPERIMENTS.keys())
    else:
        keys = [part.strip() for part in choice.split(",") if part.strip()]

    unknown = [key for key in keys if key not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Неизвестные номера экспериментов: {unknown}")

    for key in keys:
        EXPERIMENTS[key]["func"]()

    return True


def maybe_run_experiments() -> bool:
    print_header("4. Эксперименты модели")

    if not ask_yes_no("4) Запустить эксперименты модели?", default="n"):
        print("Эксперименты модели пропущены.")
        return False

    return choose_and_run_experiments()


def run_rul_analysis() -> None:
    print_header("5. Дополнительный RUL-анализ")

    RUN_LOG.clear()

    print("Будут последовательно запущены сценарии:")
    for index, step in enumerate(RUL_ANALYSIS_STEPS, start=1):
        print(f"{index}. {step['name']} — {step['script']}")

    print()
    print(
        "Для корректной работы RUL-анализа должны быть подготовлены файлы "
        "processed/*_dataset_labeled.parquet, processed/*_train_prepared.parquet, "
        "processed/*_test_prepared.parquet и результаты классификации с threshold-метриками."
    )

    for step in RUL_ANALYSIS_STEPS:
        print_header(f"RUL-анализ: {step['name']}")
        run_python_script(step["script"], step["args"])

    log_path = RESULTS_DIR / "rul_analysis_run_commands.txt"
    log_path.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    print()
    print(f"Лог команд RUL-анализа сохранен: {log_path}")


def maybe_run_rul_analysis(default: str = "n") -> bool:
    print_header("5. Анализ RUL")

    if not ask_yes_no("5) Запустить дополнительный RUL-анализ скриптами 5_1–5_4?", default=default):
        print("RUL-анализ пропущен.")
        return False

    run_rul_analysis()
    return True


def main() -> None:
    print_header("Hosokawa RUL: управляющий запуск моделей")

    ensure_project_dirs()
    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка кода:     {MODELS_DIR}")
    print(f"Датасеты:       {PROCESSED_DIR}")
    print(f"Журналы ТОИР:   {TOIR_DIR}")
    print(f"Результаты:     {RESULTS_DIR}")

    if ask_yes_no("1) Распаковать основной архив MSZ.zip?", default="n"):
        unpack_main_archive()
    else:
        print("Распаковка архива пропущена.")

    if ask_yes_no("2) Убрать полные дубли у базового и SSD parquet?", default="n"):
        remove_duplicates_for_known_parquets()
    else:
        print("Удаление дублей пропущено.")

    if ask_yes_no("3) Обработать журналы ТОИР скриптами 3_1–3_5?", default="n"):
        process_toir_journals()
    else:
        print("Обработка журналов пропущена.")

    experiments_ran = maybe_run_experiments()

    # Если эксперименты не запускались, чаще всего пользователь хочет перейти сразу к RUL-анализу.
    # Поэтому для такого случая значение по умолчанию — "y". После экспериментов RUL-анализ
    # тоже предлагается, но по умолчанию не запускается, чтобы случайно не начать долгий расчет.
    maybe_run_rul_analysis(default="n" if experiments_ran else "y")

    print_header("Готово")
    print("Работа управляющего скрипта завершена.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print()
        print("ОШИБКА: один из дочерних скриптов завершился с ошибкой.")
        print(f"Код возврата: {exc.returncode}")
        print(f"Команда: {exc.cmd}")
        sys.exit(exc.returncode)
