# Employee Salary Prediction Pipeline

AI/ML Build Week – Day 6 Assignment  
**Topic:** End-to-end ML pipeline for employee salary prediction using scikit-learn

---

## Project Structure

```
salary-pipeline-project/
├── data.csv                  # Employee salary dataset
├── train_pipeline.ipynb      # Training notebook (run this first)
├── predict_salary.ipynb      # Inference notebook
├── salary_pipeline.pkl       # Saved model (generated after training)
├── README.md
└── requirements.txt
```

---

## Setup & Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 – Train the model
Open and run all cells in `train_pipeline.ipynb`:
- Loads `data.csv`
- Preprocesses features (imputation, encoding, scaling)
- Trains 3 models: Linear Regression, Decision Tree, Random Forest
- Evaluates with MAE, RMSE, R² Score
- Saves the best model to `salary_pipeline.pkl`

### Step 2 – Predict salaries
Open and run all cells in `predict_salary.ipynb`:
- Loads `salary_pipeline.pkl`
- Predicts salaries for new employee data
- Interactive single-employee prediction cell included

---

## Pipeline Architecture

```
Input Features
     │
     ├── Numerical (Age, Years_Experience)
     │       └── SimpleImputer(median) → StandardScaler
     │
     └── Categorical (Gender, Education, Job_Title, Department)
             └── SimpleImputer(most_frequent) → OneHotEncoder
     │
     └── ColumnTransformer (combines both)
             └── Regression Model (LinearRegression / RandomForest / DecisionTree)
```

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Employee age |
| Gender | Categorical | Male / Female |
| Education | Categorical | Bachelor's / Master's / PhD |
| Job_Title | Categorical | Job role |
| Years_Experience | Numerical | Years of work experience |
| Department | Categorical | Company department |

**Target:** `Salary` (in USD)

---

## Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Coefficient of Determination

---

## Tech Stack

- Python 3.9+
- scikit-learn (Pipeline, ColumnTransformer, models)
- pandas, numpy
- matplotlib, seaborn
- joblib
