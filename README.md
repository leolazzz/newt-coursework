# Newt coursework

Небольшое совместное дообучение официального multitask Newt S (2M).

## Запуск на Kaggle

1. Загрузить `newt-coursework.zip` в приватный Kaggle Dataset.
2. Импортировать `kaggle.ipynb` в новый Notebook и подключить Dataset через Add Input.
3. Включить Internet и GPU T4. Используется одна GPU.
4. Выбрать Save Version, затем Save & Run All для фонового запуска всех ячеек.
5. После завершения скачать `newt-multitask-full-results.zip` из Output.

Четыре кодовые ячейки: копирование проекта, установка, эксперимент, вывод таблицы.
Код работает из `/kaggle/working/newt-coursework`, окружение отдельно в `/kaggle/temp`.
Ориентир по пилоту: 7-9 часов без установки. Лимит эксперимента 11 часов, включая резерв 15 минут.
Таймер проверяется между блоками, это не жёсткое завершение процесса.
Автоматического продолжения прерванного обучения нет. Повторный запуск начинает новый эксперимент.
Выходные файлы нужно сохранить: новая сессия не восстанавливает их автоматически.

## Файлы

```text
kaggle.ipynb          запуск на Kaggle
setup.sh             отдельное окружение и один checkpoint
run.py               точка входа
requirements.txt     зависимости успешного пилота
upstream/            полная неизменённая копия репозитория авторов
scripts/             подготовка runtime и загрузка весов
newt_experiment/     наш цикл эксперимента
SOURCE.json          версии исходников и checkpoint
CHANGES.md           происхождение кода и отличия
EXPERIMENT_LOG.md    краткий журнал
```

В `newt_experiment`: settings задаёт параметры; agent создаёт модель и среды;
rollout собирает эпизод; training вызывает авторское обновление;
evaluation проверяет веса; results сохраняет таблицы и архив; main задаёт порядок запусков.

## Эксперимент

- Train: walker-stand/walk/run, hopper-stand/hop, cheetah-run.
- Held-out: walker-walk-incline, spinner-spin-four. Только оценивание.
- Две ветви сбора опыта: MPC и prior. Seeds 0, 1, 2.
- На ветвь и seed: 3000 случайных и 300000 online переходов по шести задачам, 22500 обновлений.
- Checkpoint: 60k, 120k, 180k, 240k, 300k.
- Каждый checkpoint оценивается с MPC и prior. Промежуточные точки: 5 evaluation seeds, финал и baseline: 10.
- Всего 3040 evaluation эпизодов, если бюджет позволит завершить весь протокол.

Это адаптация готовой multitask-модели на знакомых задачах, а не обучение с нуля.
Held-out задачи отсутствуют в закреплённом списке soup; это узкая проверка новых вариантов задач,
не доказательство переноса в произвольную новую среду.
Наш цикл и уменьшенный бюджет отличаются от обучения статьи. Результаты не подменяют авторские CSV.

В `runs/<время>/`: сырые rollout NPZ, losses JSONL, конфигурации, checkpoint, время,
сведения о GPU, `episodes.csv`, `by-training-seed.csv`, `summary.csv`, `status.json`.
Score DMControl: сумма наград эпизода / 1000. Сначала усредняются evaluation seeds
внутри обучения, затем считается SD между training seeds. Время действия в таблице относится к целому batch сред.

[Официальный код](https://github.com/nicklashansen/newt) · [Статья](https://www.nicklashansen.com/NewtWM/) · [Веса](https://huggingface.co/nicklashansen/newt)

Репозиторий подготовлен локально. Загрузка на GitHub и Kaggle выполняется отдельно.
