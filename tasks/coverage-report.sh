cd ..
coverage run --omit="*test*" -m pytest
coverage report -m --fail-under 80
rm coverage.svg
coverage-badge -o coverage.svg