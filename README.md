# Newt coursework

Совместное дообучение multitask Newt S (2M): сбор опыта с MPC и prior,
проверка обеих ветвей в обоих режимах. Это уменьшенный эксперимент, не полное обучение статьи.

```text
train.py              полный эксперимент
inference.py          только оценивание
src/configs/          параметры YAML
src/model/            создание Newt и сред
src/trainer/          rollout, обучение, оценивание и порядок запусков
src/utils/            импорты, сохранение результатов и таймер
third_party/newt/     неизменённый авторский код
scripts/              установка и подготовка runtime
kaggle.ipynb          запуск через приватный GitHub
```

## Kaggle

1. Отправить проект в приватный GitHub-репозиторий.
2. Импортировать kaggle.ipynb. В первой ячейке указать REPOSITORY, например login/newt-coursework.
3. Создать GitHub fine-grained token только для этого репозитория: Contents -> Read-only.
   Добавить в Kaggle Secrets как GITHUB_TOKEN и разрешить доступ ноутбуку.
4. Включить Internet и T4, выбрать Save Version -> Save & Run All.

Dataset не нужен. REVISION задаёт ветку или commit; фактический SHA сохраняется в project-commit.txt.
Результат: /kaggle/working/newt-multitask-full-results.zip. Перед новой сессией сохранить Output.
Код работает в отдельном Python 3.10.22, использует одну GPU. Ориентир 7-9 часов без установки,
мягкий бюджет 11 часов с резервом 15 минут. Автоматического resume нет.

Из папки проекта после установки окружения:

```bash
python train.py
python inference.py
python inference.py checkpoint=/path/to/checkpoint.pt
```

Нужно использовать Python окружения проекта; на Kaggle его полный путь указан в ноутбуке.
Для изменения протокола отредактировать src/configs/baseline.yaml или передать Hydra override:
`python train.py experiment.train_seeds=[0]`.

## Протокол

Train: walker-stand/walk/run, hopper-stand/hop, cheetah-run.
Held-out: walker-walk-incline, spinner-spin-four, только оценивание.
Две ветви x три training seeds. На запуск: 3000 random + 300000 online переходов,
22500 updates. Checkpoint каждые 60000 переходов; 5 eval seeds на промежуточных точках,
10 на baseline и финале. Полный протокол: 3040 eval эпизодов.

В runs сохраняются raw NPZ, losses, checkpoint, конфигурация, время, GPU и таблицы.
Score DMControl = return / 1000. SD считается между training seeds после усреднения eval seeds.
Новые задачи являются узкой проверкой held-out вариантов, а не произвольной новой среды.

[Источники и изменения](docs/changes.md) · [Журнал](docs/experiments.md)

Структура по образцу [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).
Алгоритм из [официального Newt](https://github.com/nicklashansen/newt).
