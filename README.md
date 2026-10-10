# AIAP Module 5 Mini-Project 1

Dataset: place the raw data file in `data/` (not tracked by git).

## Structure

```
data/              raw dataset (gitignored)
eda.ipynb          exploratory data analysis
main.py            pipeline entry point
models/            saved model artifacts (gitignored)
reports/           metrics JSON and figures
src/               pipeline modules + config.yaml
tests/             pytest suite
```

## Setup

```bash
pip install -r requirements-dev.txt
python main.py
pytest
```
