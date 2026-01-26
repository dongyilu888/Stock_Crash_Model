.PHONY: setup data features train run clean

# Ensure Miniconda bin is in PATH
export PATH := /opt/miniconda3/bin:$(PATH)

PYTHON = python

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

run:
	$(PYTHON) -m streamlit run app/dashboard.py

pipeline: data features train

all: setup pipeline run
