# CodeAlpha - Credit Scoring Model 💳

## Overview
This project predicts an individual's **creditworthiness** using past financial data. Built as part of the **CodeAlpha Machine Learning Internship**.

## Approach
Multiple classification algorithms are trained and compared:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**

## Features Used
| Feature | Description |
|---------|-------------|
| `age` | Age of the individual |
| `annual_income` | Yearly income |
| `monthly_debt` | Monthly debt payments |
| `credit_history_length` | Credit history in years |
| `num_open_accounts` | Number of open credit accounts |
| `num_late_payments` | Number of late payments |
| `credit_utilization_ratio` | Credit utilization (0-1) |
| `num_credit_inquiries` | Number of credit inquiries |
| `loan_amount` | Requested loan amount |
| `employment_years` | Years of employment |
| `debt_to_income` | Debt-to-Income ratio (engineered) |

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix
- 5-Fold Cross-Validation

## How to Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```

## Results
The best model is selected based on ROC-AUC score. All comparison charts and the confusion matrix are saved as PNG files.

## Author
**Samah AZIZ** — CodeAlpha ML Internship
