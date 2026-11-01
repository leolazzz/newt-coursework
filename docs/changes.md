# Происхождение кода

- third_party/newt: полные неизменённые исходники авторов, версия в sources.json.
- В runtime меняются только envs/__init__.py (оставлена фабрика DMControl) и
  envs/wrappers/vectorized_multitask.py (ленивый импорт EmbeddingWrapper).
  Diff: scripts/runtime-imports.patch. Модель, planner, buffer и обновления не изменены.
- src, train.py, inference.py, scripts и kaggle.ipynb написаны для нашего эксперимента с помощью ассистента.
- agent.py сохраняет глобальные индексы soup и добавляет готовые авторские embeddings и masks для двух held-out задач.
- trainer.py использует авторский Buffer и agent.update, но наше расписание: 3000 случайных переходов,
  затем 225 updates после каждого balanced rollout из 3000 переходов. Оптимизаторы новые, новых demos нет.
- evaluator.py вызывает eval/no_grad, не вызывает update и сохраняет каждый завершённый batch.
- Оценивание идёт после обучения, по фиксированным checkpoint. Агрегация сначала внутри training seed.
- Структура src, YAML-конфигурации и точки train/inference следуют идее
  https://github.com/Blinorot/pytorch_project_template. Код этого шаблона не копировался.

Это уменьшенное дообучение готового multitask S. Batch 128, replay 12000, UTD 0.075,
compile выключен. Это отличается от протокола статьи. planner_step задаёт режим annealing,
а не число переходов нашего обучения. Флаг cfg.finetune=False нужен для полной загрузки soup;
сами веса нейросетей в обучении обновляются. OOD означает две задачи вне закреплённого списка soup.
