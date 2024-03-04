chmod +x env/bin/activate
. env/bin/activate
pip uninstall --yes rstk
python -m build --sdist --wheel
pip install -e .