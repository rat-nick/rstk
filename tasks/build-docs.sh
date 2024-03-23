cd ..
. env/bin/activate
sphinx-apidoc -o docs src
make -C docs html