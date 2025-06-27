prepare:
	@black .
	@isort .
	@mypy job
	@pylint job
	@flake8 job
	@echo Good to Go!

check:
	@black .
	@isort .
	@mypy src
	@flake8 src
	@pylint src
	@echo Good to Go!

test:
	@pytest --cov-report=term-missing --cov=src tests/

test-cov:
	@pytest --cov src tests/ --cov-report xml:coverage.xml

install:
	@pip install -r requirements.txt

clean:
	@rm -rf __pycache__