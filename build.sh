chmod +x env/bin/activate
. env/bin/activate
python setup.py sdist bdist_wheel
pip install .