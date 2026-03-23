#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = s2t-fs
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Download datasets

.PHONY: download_processed_data
download_processed_data:
# 	pip install gdown
# 	mkdir -p data/processed
	gdown 1SzQLGqwpKuLEUf_4tIzvneg9T_RBNVtv -O data/processed/voxpopuli.parquet
	gdown 1gUPWooWpyNx-mbSB-mFDUqVZtIzD8Q7U -O data/processed/librispeech.parquet
	gdown 1EzfaIOovXBY5pfxYdp9Pgq2YAXRWD50Q -O data/processed/ami.parquet
	gdown 1hpqNdUI4y_4lD2Gj3tC2QWnsUbK0lOxZ -O data/processed/common_voice.parquet




## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

## Serve documentation locally
.PHONY: docs-serve
docs-serve:
	mkdocs serve

## Build documentation
.PHONY: docs-build
docs-build:
	mkdocs build

## Deploy documentation to gh-pages locally
.PHONY: docs-deploy
docs-deploy:
	mkdocs gh-deploy --force


.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
