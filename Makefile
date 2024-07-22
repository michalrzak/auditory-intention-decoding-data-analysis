#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = auditory-intention-decoding-data-analysis
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = poetry run python

FIGURES = "reports/figures"
DUMMY_FILE = ".dummy"

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
requirements: poetry.lock
poetry.lock: pyproject.toml
	poetry install



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

RAW_FILES := $(wildcard data/raw/sub-*/ses-*/*.vhdr)
RE_REFERENCED_FILES := $(foreach file, $(RAW_FILES), $(shell echo $(file) | sed 's|data/raw/sub-[^/]*/ses-[^/]*/\(.*\).vhdr|data/processed/\1-average_reference.fif|'))
PROCESSED_FILES := $(foreach file, $(RAW_FILES), $(shell echo $(file) | sed 's|data/raw/sub-[^/]*/ses-[^/]*/\(.*\).vhdr|data/processed/\1.fif|'))

## Make Dataset
data: requirements $(RE_REFERENCED_FILES) $(PROCESSED_FILES)
data/processed/%-average_reference.fif: data/raw/sub-*/ses-*/%.vhdr
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/dataset.py $< $ true
data/processed/%.fif: data/raw/sub-*/ses-*/%.vhdr
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/dataset.py $< $@ false

#--------------------------------------------------------------------------------
ERP_PLOT_TARGET := "$(FIGURES)/erp-plots/$(DUMMY_FILE)"
ERP_PLOT_TARGET_AVERAGE_REFERENCE := "$(FIGURES)/erp-plots-average-reference/$(DUMMY_FILE)"

## Make ERP-plots
erp-plot: $(ERP_PLOT_TARGET) $(ERP_PLOT_TARGET_AVERAGE_REFERENCE)
$(ERP_PLOT_TARGET): $(PROCESSED_FILES)
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/plots/plot_erp.py $^ $@
$(ERP_PLOT_TARGET_AVERAGE_REFERENCE): $(RE_REFERENCED_FILES)
	$(PYTHON_INTERPRETER) auditory_intention_decoding_data_analysis/plots/plot_erp.py $^ $@


#--------------------------------------------------------------------------------

all: requirements data erp-plot


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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
