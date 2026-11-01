# Newt coursework

Небольшое совместное дообучение официального multitask Newt S (2M).

## Запуск на Kaggle

1. Создать пустой приватный GitHub-репозиторий `newt-coursework`, без README и .gitignore.
2. Из локальной папки проекта отправить существующий commit:

```powershell
cd C:\Users\leola\Downloads\newt\newt-coursework
git remote add origin https://github.com/YOUR_GITHUB_LOGIN/newt-coursework.git
git push -u origin main
```

В URL заменить YOUR_GITHUB_LOGIN. Для push используется обычная локальная авторизация GitHub.

3. Создать GitHub fine-grained token: доступ только к этому репозиторию,
   Repository permissions -> Contents -> Read-only.
4. Импортировать `kaggle.ipynb` в Kaggle. В настройках Secrets добавить `GITHUB_TOKEN`
   со значением токена и разрешить доступ этому ноутбуку. Сам токен в ячейки не вставлять.
5. В первой ячейке заменить REPOSITORY на `логин/newt-coursework`.
   REVISION может быть main или точным commit SHA. Фактический SHA сохраняется в project-commit.txt.
6. Включить Internet и GPU T4. Выбрать Save Version -> Save & Run All.
7. После завершения скачать `newt-multitask-full-results.zip` из Output.

Dataset не нужен. Первая ячейка получает snapshot приватного репозитория через GitHub API
по конкретному commit и распаковывает проект. Остальные устанавливают окружение,
запускают эксперимент и показывают таблицу. Токен не записывается в файлы проекта.
Для повторения прежнего запуска указать SHA из project-commit.txt в REVISION.

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
