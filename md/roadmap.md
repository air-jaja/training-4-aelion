# Project Roadmap: Data Analysis with Python and Pandas

## Objective
Analyze the `gold_dataset.csv` file by loading it into a Python notebook, sorting the data by **machine** and **time**, and performing basic exploratory steps.

---

## Key Steps

### 1. Set Up the Environment
- **Goal**: Create a Python notebook and ensure required libraries are installed.
- **Actions**:
  - Create a new Jupyter Notebook (e.g., `analysis.ipynb`).
  - Install dependencies if missing:
    ```bash
    uv install pandas jupyter
    ```
- **Verification**: Run `import pandas as pd` in a notebook cell to confirm the library is available.

---

### 2. Load the CSV File
- **Goal**: Load `gold_dataset.csv` into a Pandas DataFrame.
- **Code Snippet**:
  ```python
  import pandas as pd

  # Load the CSV file into a DataFrame
  df = pd.read_csv('gold_dataset.csv')


  # Project Roadmap: Predictive Maintenance Data Preparation

## Objective
Prepare the `gold_dataset.parquet` for predictive maintenance modeling by:
1. Loading and sorting the data by machine and time.
2. Excluding leakage columns, identifiers, and labels from features.
3. Building a binary target variable (`y`) from a selected horizon (e.g., `label_failure_next_24h`).
4. Using the provided temporal split (`split_set`: train < validation < test).
5. Imputing NaN values (median) and standardizing features for linear models.

---

## Prerequisites
- **Python 3.10+** installed.
- **uv** (Python package manager) installed. If not, install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh   sh

  