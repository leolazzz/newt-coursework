from pathlib import Path
import shutil
import difflib
import hashlib

def prepare(repo, runtime, output):
    repo, runtime, output = map(Path, (repo, runtime, output))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo / 'tdmpc2', runtime / 'tdmpc2', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(repo / 'tasks.json', runtime / 'tasks.json')
    paths = ['tdmpc2/envs/__init__.py', 'tdmpc2/envs/wrappers/vectorized_multitask.py']
    diff = []
    for rel in paths:
        path = runtime / rel
        original = path.read_text()
        modified = original
        if rel.endswith('envs/__init__.py'):
            domains = ['maniskill', 'metaworld', 'mujoco', 'box2d', 'robodesk', 'ogbench', 'pygame', 'atari']
            for domain in domains:
                line = f'from envs.{domain} import make_env as make_{domain}_env\n'
                modified = modified.replace(line, '')
            old = ('\t\tfor fn in [\n'
                   '\t\t\tmake_dm_control_env, make_maniskill_env, make_metaworld_env,\n'
                   '\t\t\tmake_mujoco_env, make_box2d_env, make_robodesk_env,\n'
                   '\t\t\tmake_ogbench_env, make_pygame_env, make_atari_env,\n'
                   '\t\t]:')
            modified = modified.replace(old, '\t\tfor fn in [make_dm_control_env]:')
        else:
            old = 'from envs.wrappers.embedding import EmbeddingWrapper\n'
            modified = modified.replace(old, '')
            old = "\tif cfg.obs == 'rgb':\n\t\tenv = EmbeddingWrapper(env)"
            modified = modified.replace(old, "\tif cfg.obs == 'rgb':\n\t\tfrom envs.wrappers.embedding import EmbeddingWrapper\n\t\tenv = EmbeddingWrapper(env)")
        path.write_text(modified)
        diff.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                         fromfile='official/' + rel, tofile='runtime/' + rel))

    (output / 'runtime-imports.patch').write_text(''.join(diff))
    (output / 'tasks-json-sha256.txt').write_text(hashlib.sha256((repo / 'tasks.json').read_bytes()).hexdigest() + '\n')
    print('Runtime ready.')

if __name__ == '__main__':
    import sys
    prepare(*sys.argv[1:])
