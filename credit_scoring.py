"""
CodeAlpha Internship - Task 1: Credit Scoring Model
====================================================
Objective: Predict an individual's creditworthiness using past financial data.
Approach: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
Author: Samah AZIZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA GENERATION (Synthetic Credit Scoring Dataset)
# ============================================================
print("=" * 60)
print("TASK 1: CREDIT SCORING MODEL")
print("=" * 60)

np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'annual_income': np.random.normal(50000, 20000, n_samples).clip(15000, 200000).astype(int),
    'monthly_debt': np.random.normal(1500, 800, n_samples).clip(0, 8000).astype(int),
    'credit_history_length': np.random.randint(0, 30, n_samples),  # years
    'num_open_accounts': np.random.randint(1, 15, n_samples),
    'num_late_payments': np.random.randint(0, 20, n_samples),
    'credit_utilization_ratio': np.random.uniform(0, 1, n_samples).round(2),
    'num_credit_inquiries': np.random.randint(0, 10, n_samples),
    'loan_amount': np.random.normal(15000, 10000, n_samples).clip(1000, 80000).astype(int),
    'employment_years': np.random.randint(0, 40, n_samples),
})

# Feature engineering: Debt-to-Income Ratio
data['debt_to_income'] = (data['monthly_debt'] * 12 / data['annual_income']).round(3)

# Generate target (creditworthy: 1 = good, 0 = bad) based on logical rules
score = (
    - 0.3 * data['num_late_payments']
    - 0.4 * data['credit_utilization_ratio']
    - 0.2 * data['debt_to_income']
    + 0.1 * data['credit_history_length']
    + 0.05 * data['employment_years']
    - 0.15 * data['num_credit_inquiries']
    + np.random.normal(0, 0.3, n_samples)
)
data['creditworthy'] = (score > score.median()).astype(int)

print(f"\n[DATA] Dataset shape: {data.shape}")
print(f"[DATA] Class distribution:\n{data['creditworthy'].value_counts()}")
print(f"\n[DATA] First 5 rows:\n{data.head()}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[0], square=True, linewidths=0.5)
axes[0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')

# Target distribution
colors = ['#e74c3c', '#2ecc71']
data['creditworthy'].value_counts().plot(kind='bar', ax=axes[1], color=colors)
axes[1].set_title('Credit Score Distribution', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['Bad Credit (0)', 'Good Credit (1)'], rotation=0)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] EDA plots saved to 'eda_analysis.png'")

# Feature importance visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features_to_plot = ['num_late_payments', 'credit_utilization_ratio', 'debt_to_income',
                    'credit_history_length', 'annual_income', 'num_credit_inquiries']

for idx, feat in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    for label, color in zip([0, 1], colors):
        subset = data[data['creditworthy'] == label]
        ax.hist(subset[feat], bins=30, alpha=0.6, color=color,
                label=f"{'Bad' if label == 0 else 'Good'}")
    ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
    ax.legend()

plt.suptitle('Feature Distribution by Creditworthiness', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Feature distributions saved to 'feature_distributions.png'")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("3. DATA PREPROCESSING")
print("=" * 60)

X = data.drop('creditworthy', axis=1)
y = data['creditworthy']

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Train set: {X_train_scaled.shape[0]} samples")
print(f"[OK] Test set:  {X_test_scaled.shape[0]} samples")
print(f"[OK] Features:  {X_train_scaled.shape[1]}")

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': auc, 'CV Mean': cv_scores.mean(),
        'y_proba': y_proba
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 5. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("5. MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k != 'y_proba'}
    for name, vals in results.items()
}).T.round(4)

print(comparison_df.to_string())

# Best model
best_model_name = comparison_df['ROC-AUC'].idxmax()
print(f"\n[BEST] Best Model: {best_model_name} (ROC-AUC: {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f})")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("6. GENERATING VISUALIZATIONS")
print("=" * 60)

# --- ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for name, vals in results.items():
    fpr, tpr, _ = roc_curve(y_test, vals['y_proba'])
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={vals['ROC-AUC']:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# --- Model comparison bar chart ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.2
colors_models = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (name, vals) in enumerate(results.items()):
    values = [vals[m] for m in metrics]
    axes[1].bar(x + i * width, values, width, label=name, color=colors_models[i], alpha=0.85)

axes[1].set_xlabel('Metrics', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(metrics)
axes[1].legend(fontsize=9)
axes[1].set_ylim(0.5, 1.05)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Model comparison plots saved to 'model_comparison.png'")

# --- Confusion Matrix for best model ---
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Bad Credit', 'Good Credit'],
            yticklabels=['Bad Credit', 'Good Credit'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Confusion matrix saved to 'confusion_matrix.png'")

# --- Feature Importance (Random Forest) ---
rf_model = models['Random Forest']
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
feature_imp.plot(kind='barh', color='#3498db', ax=ax)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Feature importance saved to 'feature_importance.png'")

# ============================================================
# 7. CLASSIFICATION REPORT (Best Model)
# ============================================================
print("\n" + "=" * 60)
print(f"7. DETAILED CLASSIFICATION REPORT — {best_model_name}")
print("=" * 60)
print(classification_report(y_test, y_pred_best, target_names=['Bad Credit', 'Good Credit']))

print("\n" + "=" * 60)
print("[OK] TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)
