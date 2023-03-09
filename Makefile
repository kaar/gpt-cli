PYTHON = python3
PYTEST = pytest

test:
	@pipenv run $(PYTEST) -v gpt/*_test.py

clean:
	rm -rf $(RESULTS)

.PHONY: test clean
