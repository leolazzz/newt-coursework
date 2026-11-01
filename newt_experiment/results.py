import json
import csv
import statistics
import zipfile
from pathlib import Path
from . import settings as S


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def export_results():
    rows = []
    for path in sorted(S.OUTPUT.rglob('evaluation.json')):
        if path.parent.name == 'baseline':
            info = dict(branch='baseline', training_seed=None, online_transitions=0, training_seconds=0)
        else:
            info = json.loads((path.parent/'info.json').read_text())
        rows.extend(dict(**info, **row) for row in json.loads(path.read_text()))
    write_csv(S.OUTPUT/'episodes.csv', rows)
    keys = ['branch','training_seed','online_transitions','task','split','mode']
    runs = []
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row['score'])
    for key, scores in groups.items():
        runs.append(dict(zip(keys,key), n_eval_episodes=len(scores),
            score_mean=statistics.mean(scores),
            eval_score_sd=statistics.stdev(scores) if len(scores)>1 else None))
    write_csv(S.OUTPUT/'by-training-seed.csv', runs)
    keys.remove('training_seed')
    groups, summary = {}, []
    for row in runs:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row['score_mean'])
    for key, scores in groups.items():
        summary.append(dict(zip(keys,key), n_runs=len(scores), score_mean=statistics.mean(scores),
            training_seed_sd=statistics.stdev(scores) if len(scores)>1 else None))
    write_csv(S.OUTPUT/'summary.csv', summary)


def bundle():
    temp = S.ARCHIVE.with_suffix('.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in S.ROOT.rglob('*'):
            rel = path.relative_to(S.ROOT)
            if path.is_file() and not any(p in {'upstream','checkpoints','__pycache__','.git'} for p in rel.parts):
                z.write(path, Path('newt-multitask-full')/rel)
    temp.replace(S.ARCHIVE)
