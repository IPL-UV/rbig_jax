.PHONY: conda format style types black test link check notebooks
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = rbig_jax

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Install Environments

conda:  ## setup a conda environment
		$(info Installing the environment)
		@printf "Creating conda environment...\n"
		${CONDA} create env create -f environment.yml
		@printf "\n\nConda environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

conda_dev:  ## setup a conda environment for development
		@printf "Creating conda dev environment...\n"
		${CONDA} create env create -f environment_dev.yml
		@printf "\n\nConda dev environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

##@ Update Environments

envupdate: ## update conda environment
		@printf "Updating conda environment...\n"
		${CONDA} env update -f environment.yml
		@printf "Conda environment updated!"
	
envupdatedev: ## update conda environment
		@printf "Updating conda dev environment...\n"
		${CONDA} env update -f environment_dev.yml
		@printf "Conda dev environment updated!"

##@ Formatting

black:  ## Format code in-place using black.
		black ${PKGROOT}/ tests/ -l 79 .

format: ## Code styling - black, isort
		black --check --diff ${PKGROOT} tests
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		isort -rc ${PKGROOT}/ tests/
		@printf "\033[1;34misort passes!\033[0m\n\n"

style:  ## Code lying - pylint
		@printf "Checking code style with flake8...\n"
		flake8 ${PKGROOT}/
		@printf "\033[1;34mPylint passes!\033[0m\n\n"
		@printf "Checking code style with pydocstyle...\n"
		pydocstyle ${PKGROOT}/
		@printf "\033[1;34mpydocstyle passes!\033[0m\n\n"

lint: format style types  ## Lint code using pydocstyle, black, pylint and mypy.
check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

##@ Type Checking

build:
	jupyter-book build jupyterbook --all

clean:
	jupyter-book clean jupyterbook
