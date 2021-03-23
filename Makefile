.PHONY: default
.DEFAULT_GOAL :=
default: install

ifndef VERBOSE
.SILENT:
endif

###############################################################################
## Dependency installation targets

.PHONY: dependencies

VENV_PATH ?= venv
VENV_DEPENDENCIES = $(VENV_PATH)/lib
VENV_ACTIVATE = . $(VENV_PATH)/bin/activate

PYTHON = $(shell which python3.7)
ifeq ($(PYTHON),)
  PYTHON := "python3"
endif

# Makes sure we have the required version ==3.7
CHECK-PYTHON-VERSION = ${PYTHON} -c 'import sys ; sys.exit(0 if (sys.version_info[0] == 3 and sys.version_info[1] == 7) else 1)' || ( \
		echo -n 'python3.7 is not the minimum required version (>= 3.7). Version found: ' ; python3.7 --version ; \
		echo '' ; \
		echo 'We recommend installing with pyenv:' ; \
		echo 'curl https://pyenv.run | bash' ; \
		echo 'Add the lines suggested by the command above to your .bashrc' ; \
		echo 'pyenv install 3.7.7 ; pyenv local 3.7.7' ; \
		echo '' ; exit 1 )


# Creates venv directory when missing
$(VENV_PATH):
	$(CHECK-PYTHON-VERSION)
	${PYTHON} -m venv $(VENV_PATH)
	touch $(VENV_PATH)

# Install venv dependencies when needed
$(VENV_DEPENDENCIES): $(VENV_PATH) requirements.txt
	$(VENV_ACTIVATE) && pip install --upgrade pip setuptools wheel
	$(VENV_ACTIVATE) && pip install -r requirements.txt --progress-bar off
	touch $(VENV_DEPENDENCIES)

dependencies: $(VENV_DEPENDENCIES)
	echo $(VENV_ACTIVATE)

################################################################################
## Clean targets

.PHONY: clean clean-engines

clean-env:
	echo "\033[1mMake: Cleaning virtual environment...\033[0m"
	rm -rf venv/
	echo "\033[1mMake: Cleaning Temporary files (/tmp/numi)...\033[0m"
	rm -rf /tmp/numi

clean: clean-env

# ###############################################################################
## Build and install targets

.PHONY: test lint build

test: dependencies
	$(VENV_ACTIVATE) && python -m pytest tests

lint: dependencies
	$(VENV_ACTIVATE) && pylint tests && python -m flake8 tests/

install: test lint
	echo "\033[0;32m[BUILD SUCCESSFUL]\033[0m"

## Clean targets

.PHONY: clean-dependencies

clean-dependencies:
	rm -rf $(VENV_PATH)

