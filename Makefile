.PHONY: setup-venv clean

VENV_PATH := ~/envs/dlp
PIP := $(VENV_PATH)/bin/pip
PYTHON_VENV := $(VENV_PATH)/bin/python

setup-venv: $(VENV_PATH)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV_PATH)/bin/activate:
	python3 -m venv $(VENV_PATH)

clean:
	rm -rf $(VENV_PATH)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete