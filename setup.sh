set -e
project="$(cd "$(dirname "$0")" && pwd)"
env_dir=/kaggle/temp/newt-isolated/python-env
export PYTHONPATH=/kaggle/temp/newt-isolated/bootstrap
export UV_CACHE_DIR=/kaggle/temp/newt-isolated/cache
export UV_PYTHON_INSTALL_DIR=/kaggle/temp/newt-isolated/python
export UV_PYTHON_BIN_DIR=/kaggle/temp/newt-isolated/bin
python -m pip install --target /kaggle/temp/newt-isolated/bootstrap --upgrade uv==0.13.5
if [ ! -x "$env_dir/bin/python" ]; then
    python -m uv venv --python 3.10.22 --seed "$env_dir"
fi
python -m uv pip install --python "$env_dir/bin/python" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python "$env_dir/bin/python" -r "$project/requirements.txt"
python -m uv pip freeze --python "$env_dir/bin/python" > "$project/environment.txt"
unset PYTHONPATH
printf "%s\n" 1d3fc058b81ddf8d36a5457c29ea407dd374c1b8 > "$project/source-commit.txt"
"$env_dir/bin/python" "$project/scripts/prepare_runtime.py" "$project/upstream" /kaggle/temp/newt-isolated/runtime "$project"
"$env_dir/bin/python" "$project/scripts/download_weights.py"
