# Universal ML Dashboard

A Streamlit-based dashboard for exploring datasets, performing data cleaning and exploratory data analysis (EDA), and training machine learning models through a user interface.

## Features

- **Dataset loading:** Supports CSV, Excel (`.xlsx`), and JSON files.
- **Dataset inspection:** View dataset dimensions, columns, data types, missing values, and duplicate rows.
- **Data cleaning:** Inspect and handle common data-quality issues.
- **Exploratory data analysis:** Analyze dataset characteristics and relationships.
- **Machine learning:** Train classification and regression models.
- **Model evaluation:** Review task-specific metrics and classification confusion matrices.
- **Automated preprocessing:** Handle numerical and categorical features through scikit-learn pipelines.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- OpenPyXL
- Pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Aryn1102/upskillcampus.git
cd upskillcampus
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (Git Bash):

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run UI/streamlit/app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

## Running Tests

Run the automated test suite from the repository root:

```bash
python -m pytest -q
```

## Project Status

The dashboard currently supports dataset inspection, exploratory analysis, classification and regression training, and model evaluation. Development and testing are ongoing.
