# Происхождение кода

`upstream/` целиком взят через git archive из nicklashansen/newt,
commit 1d3fc058b81ddf8d36a5457c29ea407dd374c1b8. Файлы внутри не редактировались.
Включены также авторские README, CSV, assets, docker и остальные среды;
авторские CSV не используются как наши результаты.

Для выполнения scripts/prepare_runtime.py копирует tdmpc2 и tasks.json в отдельный runtime.
В нём меняются только:

1. tdmpc2/envs/__init__.py: убраны импорты необязательных доменов и их фабрики из списка.
   Оставлена фабрика DMControl. Это позволяет не ставить остальные симуляторы.
2. tdmpc2/envs/wrappers/vectorized_multitask.py: импорт EmbeddingWrapper перенесён
   внутрь условия obs == rgb. Для state-наблюдений не нужны зависимости обработки изображений.

Точный diff сохраняется в runtime-imports.patch. Архитектура, planner, функции потерь,
обновление параметров и Buffer остаются авторскими, включая replay prefetch.

Всё вне upstream написано для этого эксперимента с помощью ассистента:

- scripts/download_weights.py: один multitask checkpoint с закреплённой HF revision.
- setup.sh, requirements.txt: отдельный Python 3.10.22, torch 2.8.0 CUDA 12.8 и зависимости пилота.
- newt_experiment/official.py: отключение телеметрии и импорты авторских классов.
- settings.py: задачи, seeds, бюджет и точки сохранения.
- agent.py: загрузка soup-S с исходными индексами всех задач, затем добавление двух
  фиксированных авторских text_embedding и action masks из tasks.json для held-out задач.
  Нейросети не меняются, embeddings не обучаются. cfg.finetune=False нужен для загрузки
  полного soup, хотя сам эксперимент является дообучением его параметров.
- rollout.py: авторский agent выбирает действие, авторская среда делает шаг.
  eval/no_grad отключают градиенты при сборе опыта и оценивании. Для обучения действие
  исследовательское; при оценивании prior используется среднее распределения.
  При autoreset сохраняется final_observation завершённого эпизода.
- training.py: balanced rollout по шести задачам, упаковка для авторского Buffer,
  225 вызовов agent.update после каждого блока 3000 переходов. Случайный warmup без updates.
  Каждый запуск начинает с одного pretrained checkpoint и новых оптимизаторов.
- evaluation.py: отдельный агент без agent.update, два режима действий и фиксированные seeds.
  Завершённые batches записываются сразу, чтобы сохранить частичное оценивание.
- results.py: сырые эпизоды, усреднение внутри seed, SD между training seeds, архив.
- utils.py: JSON и таймер. main.py: baseline, шесть обучений и их проверки.
- run.py и kaggle.ipynb: точки запуска. В ноутбуке нет скопированных авторских ячеек.

Авторский Trainer.train не используется: своё расписание нужно для явного сравнения
сбора опыта с planner и без него. Обе ветви обучают одну и ту же архитектуру и используют
одни функции потерь. Это не сравнение двух разных алгоритмов обучения.
У нас batch 128, replay 12000, UTD 0.075, нет новых demonstrations и torch.compile выключен.
Оценивание начинается после завершения соответствующего обучения, без выбора лучшего checkpoint.
PLAN_STEP=100000000 фиксирует режим planner после начального annealing;
это не число переходов нашего обучения и не подтверждённый training step исходных весов.

Kaggle bootstrap получает приватный GitHub snapshot через API с токеном из Kaggle Secrets.
Сначала main/REVISION разрешается в точный SHA; он записывается в project-commit.txt.
Git clone и subprocess для загрузки не используются, Dataset не требуется.
