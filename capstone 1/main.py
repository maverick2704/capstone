import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance


# ============================================================
# PREREQUISITES - CREATE THE CUSTOMER CHURN DATASET
# ============================================================

RNG = 42
rng = np.random.default_rng(RNG)
N = 4000

# Create the main numeric signal
X_num, y = make_classification(
    n_samples=N,
    n_features=6,
    n_informative=5,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.76, 0.24],
    flip_y=0.13,
    class_sep=0.85,
    random_state=RNG
)

# Standardize before converting to realistic customer values
Xs = (X_num - X_num.mean(0)) / X_num.std(0)

df = pd.DataFrame()

# Numeric customer features
df["tenure_months"] = np.clip(
    Xs[:, 4] * 12 + 34, 1, 72
).round().astype(int)

df["monthly_charges"] = np.clip(
    Xs[:, 0] * 13 + 72, 18, 130
).round(2)

df["support_tickets"] = np.clip(
    Xs[:, 2] * 1.7 + 3, 0, 25
).round().astype(int)

df["avg_latency_ms"] = np.clip(
    Xs[:, 3] * 35 + 120, 20, 400
).round().astype(int)

df["addons_count"] = np.clip(
    Xs[:, 1] * 1.3 + 4, 0, 8
).round().astype(int)

df["days_since_login"] = np.clip(
    Xs[:, 4] * 8 + 16 + rng.normal(0, 4, N),
    0,
    90
).round().astype(int)

df["total_charges"] = (
    df["monthly_charges"] * df["tenure_months"]
).round(2)


# Create categorical features depending on churn class
def by_class(p_churn, p_stay, choices):
    out = np.empty(N, dtype=object)

    m = (y == 1)

    out[m] = rng.choice(
        choices,
        m.sum(),
        p=p_churn
    )

    out[~m] = rng.choice(
        choices,
        (~m).sum(),
        p=p_stay
    )

    return out


df["contract_type"] = by_class(
    [0.75, 0.18, 0.07],
    [0.34, 0.35, 0.31],
    ["Month-to-month", "One year", "Two year"]
)

df["payment_method"] = by_class(
    [0.55, 0.20, 0.15, 0.10],
    [0.25, 0.30, 0.25, 0.20],
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

df["internet_service"] = by_class(
    [0.65, 0.28, 0.07],
    [0.40, 0.35, 0.25],
    ["Fiber optic", "DSL", "None"]
)

df["has_dependents"] = by_class(
    [0.82, 0.18],
    [0.60, 0.40],
    ["No", "Yes"]
)

# Deliberately create a leakage feature
df["exit_survey_sent"] = np.where(
    y == 1,
    rng.binomial(1, 0.97, N),
    rng.binomial(1, 0.02, N)
)

# Target
df["churn"] = y

print("dataset shape:", df.shape)
print("churn rate:", round(df["churn"].mean(), 3))
print(df.head())


# ============================================================
# STEP 1 - FRAME THE PROBLEM AND EXPLORE THE DATA
# ============================================================

# Check the class balance
print("\nCLASS BALANCE")
print(df["churn"].value_counts(normalize=True).round(3), "\n")

# Check numeric differences between stayers and churners
num_cols = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "avg_latency_ms",
    "addons_count",
    "days_since_login",
    "total_charges"
]

print("NUMERIC FEATURE MEANS BY CHURN")
print(df.groupby("churn")[num_cols].mean().round(1), "\n")

# Check churn rate by contract type
print("CHURN RATE BY CONTRACT TYPE")
print(
    df.groupby("contract_type")["churn"]
      .mean()
      .round(3)
      .sort_values(ascending=False)
)

# Visualise tenure and contract type against churn
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

for k, lbl in [(0, "stay"), (1, "churn")]:
    ax[0].hist(
        df.loc[df.churn == k, "tenure_months"],
        bins=25,
        alpha=0.6,
        label=lbl
    )

ax[0].set_title("Tenure by outcome")
ax[0].set_xlabel("tenure_months")
ax[0].legend()

df.groupby("contract_type")["churn"].mean().sort_values().plot.bar(
    ax=ax[1]
)

ax[1].set_title("Churn rate by contract")
ax[1].set_ylabel("churn rate")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 2 - SPLIT TRAIN / VALIDATION / TEST AND FIND LEAKAGE
# ============================================================

# Look for suspiciously high correlations with the target
leak_scores = (
    df.corr(numeric_only=True)["churn"]
      .drop("churn")
      .abs()
      .sort_values(ascending=False)
)

print("\nABSOLUTE CORRELATION WITH CHURN")
print(leak_scores.round(3), "\n")

# Features above 0.9 correlation are treated as suspicious leakage
suspects = leak_scores[leak_scores > 0.9].index.tolist()

print("LEAK suspects (corr > 0.9):", suspects)

# Remove leakage columns
df_clean = df.drop(columns=suspects)

# Separate features and target
X = df_clean.drop(columns=["churn"])
y = df_clean["churn"]

# 70% train and 30% temporary set
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RNG
)

# Split temporary set equally into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=0.50,
    stratify=y_tmp,
    random_state=RNG
)

print("\nSHAPES")
print("train:", X_train.shape)
print("validation:", X_val.shape)
print("test:", X_test.shape)

print(
    "churn rate train/val/test:",
    round(y_train.mean(), 3),
    round(y_val.mean(), 3),
    round(y_test.mean(), 3)
)


# ============================================================
# STEP 3 - PREPROCESSING + SCALING INSIDE A PIPELINE
# ============================================================

# Automatically identify numeric and categorical columns
num_features = X_train.select_dtypes("number").columns.tolist()

cat_features = (
    X_train.select_dtypes("object").columns.tolist()
)

print("\nnumeric:", num_features)
print("categorical:", cat_features)

# Apply scaling to numbers and one-hot encoding to categories
pre = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        num_features
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        cat_features
    )
])

# Fit preprocessing only on the training data
Z = pre.fit_transform(X_train)

print("\ntransformed matrix:", Z.shape)
print(
    "feature count after one-hot:",
    len(pre.get_feature_names_out())
)


# ============================================================
# STEP 4 - TRAIN AND COMPARE FOUR MODEL FAMILIES
# ============================================================

models = {
    "LogReg": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=RNG,
        class_weight="balanced"
    ),

    "GradBoost": GradientBoostingClassifier(
        random_state=RNG
    ),

    "SVM_RBF": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RNG
    )
}

# Same 5-fold split is used for each model
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RNG
)

results = {}

print("\nMODEL COMPARISON")

for name, clf in models.items():

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    results[name] = scores

    print(
        f"{name:13s} ROC-AUC = "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )


# ============================================================
# STEP 5 - TUNE GRADIENT BOOSTING WITH GRIDSEARCHCV
# ============================================================

gb_pipe = Pipeline([
    ("pre", pre),
    (
        "clf",
        GradientBoostingClassifier(
            random_state=RNG
        )
    )
])

# Parameters to test
param_grid = {
    "clf__n_estimators": [150, 300],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3]
}

grid = GridSearchCV(
    gb_pipe,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nbest params:", grid.best_params_)
print(
    "best CV ROC-AUC:",
    round(grid.best_score_, 3)
)

# Best model found by GridSearchCV
best_model = grid.best_estimator_

# Check on validation set
val_prob = best_model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(
    y_val,
    val_prob
)

print(
    "validation ROC-AUC:",
    round(val_auc, 3)
)


# ============================================================
# STEP 6 - FINAL TEST EVALUATION + THRESHOLD SELECTION
# ============================================================

# Test set is used only after model selection and tuning
proba = best_model.predict_proba(X_test)[:, 1]

# Default threshold
pred_050 = (proba >= 0.50).astype(int)

print(
    "\nTEST ROC-AUC:",
    round(roc_auc_score(y_test, proba), 3),
    "\n"
)

print(
    classification_report(
        y_test,
        pred_050,
        target_names=["stay", "churn"],
        digits=3
    )
)

# Display confusion matrix at 0.50 threshold
ConfusionMatrixDisplay(
    confusion_matrix(
        y_test,
        pred_050
    ),
    display_labels=["stay", "churn"]
).plot(values_format="d")

plt.title("Confusion matrix @ threshold 0.50")
plt.show()


# Compare different decision thresholds
print(
    f"{'thr':>5} "
    f"{'precision':>10} "
    f"{'recall':>8} "
    f"{'f1':>6}"
)

for t in [0.50, 0.40, 0.35, 0.30, 0.25]:

    p = (proba >= t).astype(int)

    print(
        f"{t:5.2f} "
        f"{precision_score(y_test, p):10.3f} "
        f"{recall_score(y_test, p):8.3f} "
        f"{f1_score(y_test, p):6.3f}"
    )


# ============================================================
# STEP 7 - INTERPRET THE MODEL
# ============================================================

# Permutation importance for the tuned Gradient Boosting model
perm = permutation_importance(
    best_model,
    X_val,
    y_val,
    n_repeats=8,
    random_state=RNG,
    scoring="roc_auc"
)

imp = pd.Series(
    perm.importances_mean,
    index=X.columns
).sort_values(ascending=False)

print(
    "\nPermutation importance "
    "(ROC-AUC drop when shuffled):"
)

print(
    imp.round(4),
    "\n"
)


# Train an interpretable Logistic Regression model
logreg = Pipeline([
    ("pre", pre),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

logreg.fit(
    X_train,
    y_train
)

# Extract Logistic Regression coefficients
coef = pd.Series(
    logreg.named_steps["clf"].coef_[0],
    index=logreg.named_steps[
        "pre"
    ].get_feature_names_out()
)

print(
    "Most churn-DRIVING (positive coef):"
)

print(
    coef.sort_values().tail(4).round(2),
    "\n"
)

print(
    "Most PROTECTIVE (negative coef):"
)

print(
    coef.sort_values().head(4).round(2)
)

# Plot permutation importance
imp.sort_values().plot.barh(
    figsize=(7, 4),
    title="Permutation importance"
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 8 - FINAL SCORECARD
# ============================================================

summary = {
    "primary_metric":
        "recall on churn (ROC-AUC for selection)",

    "chosen_model":
        "GradientBoosting (tuned)",

    "test_roc_auc":
        round(
            roc_auc_score(
                y_test,
                proba
            ),
            3
        ),

    "operating_threshold":
        0.30
}

# Predictions using selected business threshold
p30 = (proba >= 0.30).astype(int)

summary["recall_at_0.30"] = round(
    recall_score(
        y_test,
        p30
    ),
    3
)

summary["precision_at_0.30"] = round(
    precision_score(
        y_test,
        p30
    ),
    3
)

summary["top_drivers"] = (
    imp.head(3).index.tolist()
)

print("\nFINAL SCORECARD")

for k, v in summary.items():
    print(f"{k:22s}: {v}")