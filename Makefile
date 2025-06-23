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
	@pytest --cov job

test-cov:
	@pytest --cov job --cov-report xml:coverage.xml
.PHONY: docs