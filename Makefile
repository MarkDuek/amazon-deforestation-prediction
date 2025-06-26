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
	@mypy job
	@flake8 job
	@pylint job
	@echo Good to Go!

test:
	@pytest --cov-report=term-missing --cov=src tests/

test-cov:
	@pytest --cov src tests/ --cov-report xml:coverage.xml
.PHONY: docs