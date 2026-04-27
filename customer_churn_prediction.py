# ============================================================
#   Customer Churn Prediction using Machine Learning
#   Internship Project - Codec Technologies
#   Author: Harshitha T N
# ============================================================

# ── STEP 1: Install required libraries (run this in terminal) ──
# pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

print("=" * 55)
print("   Customer Churn Prediction - Machine Learning Project")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 2: Generate a Realistic Dataset
# (In real projects, use: df = pd.read_csv('your_data.csv'))
# ─────────────────────────────────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'CustomerID'      : range(1, n + 1),
    'Gender'          : np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen'   : np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'Tenure'          : np.random.randint(1, 73, n),           # months
    'MonthlyCharges'  : np.round(np.random.uniform(18, 120, n), 2),
    'TotalCharges'    : np.round(np.random.uniform(18, 8700, n), 2),
    'Contract'        : np.random.choice(
                            ['Month-to-month', 'One year', 'Two year'], n,
                            p=[0.55, 0.24, 0.21]),
    'InternetService' : np.random.choice(
                            ['DSL', 'Fiber optic', 'No'], n,
                            p=[0.34, 0.44, 0.22]),
    'PaymentMethod'   : np.random.choice(
                            ['Electronic check', 'Mailed check',
                             'Bank transfer', 'Credit card'], n),
    'TechSupport'     : np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71]),
    'OnlineSecurity'  : np.random.choice(['Yes', 'No'], n, p=[0.28, 0.72]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
    'NumSupportCalls' : np.random.randint(0, 6, n),
})

# Simulate churn realistically
churn_prob = (
    0.30 * (df['Contract'] == 'Month-to-month').astype(int) +
    0.20 * (df['Tenure'] < 12).astype(int) +
    0.15 * (df['InternetService'] == 'Fiber optic').astype(int) +
    0.10 * (df['TechSupport'] == 'No').astype(int) +
    0.05 * (df['NumSupportCalls'] > 3).astype(int) +
    np.random.uniform(0, 0.20, n)
)
df['Churn'] = (churn_prob > 0.40).astype(int)

print(f"\n📦 Dataset Created: {df.shape[0]} customers, {df.shape[1]} features")
print(f"\n📊 Churn Distribution:")
churn_counts = df['Churn'].value_counts()
print(f"   Not Churned (0): {churn_counts[0]} ({churn_counts[0]/n*100:.1f}%)")
print(f"   Churned     (1): {churn_counts[1]} ({churn_counts[1]/n*100:.1f}%)")

# ─────────────────────────────────────────
# STEP 3: Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────
print("\n🔍 Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis - Customer Churn", fontsize=15, fontweight='bold')

# Plot 1: Churn count
axes[0,0].bar(['Not Churned', 'Churned'], churn_counts.values,
               color=['steelblue', 'tomato'], edgecolor='black', width=0.5)
axes[0,0].set_title("Churn Count")
axes[0,0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Churn by Contract Type
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
axes[0,1].bar(contract_churn.index, contract_churn.values,
               color=['#e74c3c','#3498db','#2ecc71'], edgecolor='black')
axes[0,1].set_title("Churn Rate by Contract Type")
axes[0,1].set_ylabel("Churn Rate (%)")
axes[0,1].tick_params(axis='x', rotation=15)

# Plot 3: Tenure distribution by churn
df[df['Churn'] == 0]['Tenure'].hist(ax=axes[0,2], alpha=0.6,
                                     bins=20, label='Not Churned', color='steelblue')
df[df['Churn'] == 1]['Tenure'].hist(ax=axes[0,2], alpha=0.6,
                                     bins=20, label='Churned', color='tomato')
axes[0,2].set_title("Tenure Distribution by Churn")
axes[0,2].set_xlabel("Tenure (months)")
axes[0,2].legend()

# Plot 4: Monthly Charges by Churn
axes[1,0].boxplot([df[df['Churn']==0]['MonthlyCharges'],
                    df[df['Churn']==1]['MonthlyCharges']],
                   labels=['Not Churned','Churned'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[1,0].set_title("Monthly Charges vs Churn")
axes[1,0].set_ylabel("Monthly Charges ($)")

# Plot 5: Churn by Internet Service
internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
axes[1,1].bar(internet_churn.index, internet_churn.values,
               color=['#9b59b6','#e67e22','#1abc9c'], edgecolor='black')
axes[1,1].set_title("Churn Rate by Internet Service")
axes[1,1].set_ylabel("Churn Rate (%)")

# Plot 6: Support Calls vs Churn
support_churn = df.groupby('NumSupportCalls')['Churn'].mean() * 100
axes[1,2].plot(support_churn.index, support_churn.values,
               marker='o', color='tomato', linewidth=2, markersize=8)
axes[1,2].set_title("Support Calls vs Churn Rate")
axes[1,2].set_xlabel("Number of Support Calls")
axes[1,2].set_ylabel("Churn Rate (%)")
axes[1,2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("eda_analysis.png", dpi=100)
plt.show()
print("✅ EDA charts saved as 'eda_analysis.png'")

# ─────────────────────────────────────────
# STEP 4: Data Preprocessing
# ─────────────────────────────────────────
print("\n⚙️  Preprocessing Data...")

df_model = df.copy()
df_model.drop('CustomerID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

# Features and target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(f"   Features         : {X_train.shape[1]}")

# ─────────────────────────────────────────
# STEP 5: Train Multiple ML Models
# ─────────────────────────────────────────
print("\n🤖 Training Models...")

models = {
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'          : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc     = accuracy_score(y_test, y_pred) * 100
    auc     = roc_auc_score(y_test, y_proba) * 100
    cv_acc  = cross_val_score(model, X_train_sc, y_train, cv=5,
                               scoring='accuracy').mean() * 100

    results[name] = {
        'model'   : model,
        'y_pred'  : y_pred,
        'y_proba' : y_proba,
        'accuracy': acc,
        'auc'     : auc,
        'cv_acc'  : cv_acc,
    }
    print(f"   ✅ {name:25s} | Acc: {acc:.2f}% | AUC: {auc:.2f}% | CV: {cv_acc:.2f}%")

# Best model
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} (AUC: {best['auc']:.2f}%)")

# ─────────────────────────────────────────
# STEP 6: Model Comparison Chart
# ─────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
aucs        = [results[m]['auc']      for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)',
                color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, aucs,       width, label='AUC Score (%)',
                color='orange',    edgecolor='black')
ax.set_xlabel("Model")
ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison - Accuracy vs AUC Score", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10)
ax.set_ylim(50, 105)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100)
plt.show()
print("✅ Model comparison saved as 'model_comparison.png'")

# ─────────────────────────────────────────
# STEP 7: Best Model - Confusion Matrix
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned','Churned'],
            yticklabels=['Not Churned','Churned'])
plt.title(f"Confusion Matrix - {best_name}", fontweight='bold')
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix_churn.png", dpi=100)
plt.show()
print("✅ Confusion matrix saved as 'confusion_matrix_churn.png'")

# ─────────────────────────────────────────
# STEP 8: ROC Curve (All Models)
# ─────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ['steelblue', 'tomato', 'green']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.1f}%)", color=color, lw=2)

plt.plot([0,1],[0,1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison", fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=100)
plt.show()
print("✅ ROC curves saved as 'roc_curves.png'")

# ─────────────────────────────────────────
# STEP 9: Feature Importance
# ─────────────────────────────────────────
rf_model = results['Random Forest']['model']
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp  = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Feature Importance (Random Forest)", fontweight='bold')
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.show()
print("✅ Feature importance saved as 'feature_importance.png'")

# ─────────────────────────────────────────
# STEP 10: Classification Report
# ─────────────────────────────────────────
print(f"\n📊 Classification Report - {best_name}:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Not Churned', 'Churned']))

# ─────────────────────────────────────────
# STEP 11: Predict on New Customers
# ─────────────────────────────────────────
print("\n🔮 Predicting Churn for New Customers...")

# Example new customers (feature order must match training)
# Features: Gender, SeniorCitizen, Tenure, MonthlyCharges, TotalCharges,
#            Contract, InternetService, PaymentMethod, TechSupport,
#            OnlineSecurity, PaperlessBilling, NumSupportCalls

new_customers = pd.DataFrame({
    'Gender'          : [1, 0],       # 1=Male, 0=Female
    'SeniorCitizen'   : [0, 1],
    'Tenure'          : [2, 36],
    'MonthlyCharges'  : [95.0, 45.0],
    'TotalCharges'    : [190.0, 1620.0],
    'Contract'        : [0, 2],       # 0=Month-to-month, 2=Two year
    'InternetService' : [1, 0],       # 1=Fiber optic, 0=DSL
    'PaymentMethod'   : [1, 3],
    'TechSupport'     : [1, 0],
    'OnlineSecurity'  : [1, 0],
    'PaperlessBilling': [1, 0],
    'NumSupportCalls' : [4, 1],
})

new_scaled  = scaler.transform(new_customers)
preds       = best['model'].predict(new_scaled)
proba       = best['model'].predict_proba(new_scaled)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, proba)):
    label = "⚠️  WILL CHURN" if pred == 1 else "✅ Will NOT Churn"
    print(f"   Customer {i+1}: {label}  (Probability: {prob*100:.1f}%)")

print("\n" + "=" * 55)
print("  ✅ Project 2 Complete!")
print(f"  Best Model   : {best_name}")
print(f"  Test Accuracy: {best['accuracy']:.2f}%")
print(f"  AUC Score    : {best['auc']:.2f}%")
print("=" * 55)
