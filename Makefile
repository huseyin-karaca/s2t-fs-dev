#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = s2t-fs
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

# Hydra overrides. Pass any hydra CLI args via OVERRIDES, e.g.:
#   make run OVERRIDES="data=voxpopuli mode=multi model_group=full_benchmark"
#   make run OVERRIDES="+experiment=ami_margin_v1"
OVERRIDES ?=

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync


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
	gdown 1SzQLGqwpKuLEUf_4tIzvneg9T_RBNVtv -O data/processed/voxpopuli.parquet
	gdown 1gUPWooWpyNx-mbSB-mFDUqVZtIzD8Q7U -O data/processed/librispeech.parquet
	gdown 1EzfaIOovXBY5pfxYdp9Pgq2YAXRWD50Q -O data/processed/ami.parquet
	gdown 1hpqNdUI4y_4lD2Gj3tC2QWnsUbK0lOxZ -O data/processed/common_voice.parquet


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	uv sync
	@echo ">>> venv created. Activate with:\nsource .venv/bin/activate"


#################################################################################
# EXPERIMENT COMMANDS                                                           #
#################################################################################

## Run single-model HPT              (e.g. make run-single OVERRIDES="data=ami model_group=fastt_boosted_only")
.PHONY: run-single
run-single:
	$(PYTHON_INTERPRETER) -m s2t_fs.experiment mode=single $(OVERRIDES)

## Run multi-model comparison        (e.g. make run-comparison OVERRIDES="data=ami")
.PHONY: run-comparison
run-comparison:
	$(PYTHON_INTERPRETER) -m s2t_fs.experiment mode=multi $(OVERRIDES)

## Run margin optimization study     (e.g. make run-margin OVERRIDES="+experiment=ami_margin_v1")
.PHONY: run-margin
run-margin:
	$(PYTHON_INTERPRETER) -m s2t_fs.experiment mode=margin $(OVERRIDES)

## Run experiment with current defaults  (compose via OVERRIDES or +experiment=<name>)
.PHONY: run
run:
	$(PYTHON_INTERPRETER) -m s2t_fs.experiment $(OVERRIDES)


#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

## Serve documentation locally
.PHONY: docs-serve
docs-serve:
	mkdocs serve --config-file docs/mkdocs.yml

## Build documentation
.PHONY: docs-build
docs-build:
	mkdocs build --config-file docs/mkdocs.yml

## Deploy documentation to gh-pages locally
.PHONY: docs-deploy
docs-deploy:
	mkdocs gh-deploy --force --config-file docs/mkdocs.yml


#################################################################################
# VAST.AI COMMANDS                                                              #
#################################################################################

# Override any of these from the CLI: make vast-launch GPU=A100_SXM4 MAX_PRICE=0.80
GPU       ?= RTX_3090
MAX_PRICE ?= 0.30
DISK      ?= 30
INSTANCE  ?=

## Launch a vast.ai GPU instance with the pre-built Docker image (SSH-ready)
.PHONY: vast-launch
vast-launch:
	bash scripts/vast_launch.sh --gpu $(GPU) --max-price $(MAX_PRICE) --disk $(DISK)

## Print the best matching vast.ai offer without creating an instance
.PHONY: vast-dry-run
vast-dry-run:
	bash scripts/vast_launch.sh --gpu $(GPU) --max-price $(MAX_PRICE) --dry-run

## List your active vast.ai instances
.PHONY: vast-list
vast-list:
	vastai show instances

## Destroy a specific instance (INSTANCE=<id>) or all instances
.PHONY: vast-stop
vast-stop:
	@if [ -n "$(INSTANCE)" ]; then \
		vastai destroy instance $(INSTANCE); \
	else \
		vastai show instances --raw | jq -r '.[].id' | xargs -I{} vastai destroy instance {}; \
	fi


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

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
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
