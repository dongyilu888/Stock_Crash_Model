.PHONY: setup data features train run clean

# Hardcoded path to Miniconda environment
PYTHON = /opt/miniconda3/bin/python

# Ensure Miniconda bin is in PATH
export PATH := /opt/miniconda3/bin:$(PATH)

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m data.fetch_data
	$(PYTHON) -m data.process_data

features:
	$(PYTHON) -m features.build_features

train:
	$(PYTHON) -m model.train
	$(PYTHON) -m model.train_gbm
	$(PYTHON) -m model.train_survival

run:
	$(PYTHON) -m streamlit run app/dashboard.py --server.port 8501

stop:
	-@lsof -ti:8501 | xargs kill -9 2>/dev/null || true

pipeline: data features train

debug-env:
	@echo "PATH: $(PATH)"
	@which python
	@python --version
	@which streamlit || echo "streamlit not in path"
	@python -m streamlit --version

all: setup pipeline run
