# %% [markdown]
# # Dataset loading



# %%
df = pd.read_csv("../dataset/data.csv")
# info about set
print(df.shape)
print("-----------")
df.head()

# %% [markdown]
# # Preprocessing

# %% [markdown]
# Data Preprocessing Tasks
# - Data Cleaning: Fill in missing values, smooth noisy data, identify or remove outliers and noisy
# data, and resolve inconsistencies
# - Data integration: Integrate data with multiple databases or files.
# - Data transformation: Data normalization and aggregation
# - Data reduction: Data reduction in volume but produces the same or similar analytical results
# - Data discretization (for numerical data)

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# ### Missing and duplicate values

# %%
na = (df.isna().sum().to_frame("n_missing").assign(pct=lambda x: (100 * x["n_missing"] / len(df)).round(2)).sort_values(["pct", "n_missing"], ascending=False))

print("Duplicated (full-row):", df.duplicated().sum())
print("-----------------------")
print("\nTop-20 missingness:")
display(na.head(10))

# %% [markdown]
# * Drop Alchol Consumption 42% missing

# %%
df = df.drop(columns='alcohol_consumption', errors="ignore")
df.shape[1]

# %% [markdown]
# * Exploring reminaing data types of missing data

# %%
# Unique categories and counts
for col in ["caffeine_intake", "exercise_type", "gene_marker_flag"]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False).head(20))  # top 20 categories including NaN
    print("n_unique:", df[col].nunique(dropna=True))

# %% [markdown]
# * Filling with Unkown

# %%
df["caffeine_intake"] = df["caffeine_intake"].fillna("Unknown")
df["exercise_type"] = df["exercise_type"].fillna("None")


# %%
# Unique categories and counts
for col in ["caffeine_intake", "exercise_type"]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False).head(20))  # top 20 categories including NaN
    print("n_unique:", df[col].nunique(dropna=True))

# %% [markdown]
# * Gene marker flag

# %%
# df['gene_marker_flag'] = df['gene_marker_flag'].fillna(0) - doesnt make a signifciant difference
df["gene_marker_flag"].value_counts(dropna=False)

# %%
df = df.drop(columns='gene_marker_flag', errors="ignore")
df.shape[1]

# %%
# Unique categories and counts
for col in ["caffeine_intake", "exercise_type"]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False).head(20))  # top 20 categories including NaN
    print("n_unique:", df[col].nunique(dropna=True))

# %% [markdown]
# * Drop income - beacuse it is unclear (anual etc)

# %%
df = df.drop(columns='income', errors="ignore")
df.shape[1]

# %% [markdown]
# * Learns a small regression model for each column with missing values, using the other columns as predictors, and iterates until convergence.


num_cols_na = ["insulin","heart_rate","daily_steps","blood_pressure", ]

imp = IterativeImputer(
    estimator=BayesianRidge(),    
    max_iter=10,                  
    sample_posterior=False,       
    random_state=42
)

df[num_cols_na] = imp.fit_transform(df[num_cols_na])

# %% [markdown]
# #### Rechecking Missing values

# %%
na = (df.isna().sum().to_frame("n_missing").assign(pct=lambda x: (100 * x["n_missing"] / len(df)).round(2)).sort_values(["pct", "n_missing"], ascending=False))

print("Duplicated (full-row):", df.duplicated().sum())
print("-----------------------")
print("\nTop-20 missingness:")
display(na.head(10))

# %% [markdown]
# ### EDA

# %%
df.info()

# %% [markdown]
# Dropping unrelated and repeated colmuns

# %%
df[['bmi','bmi_estimated','bmi_scaled','bmi_corrected']].sample(5)

# %%
plt.plot(df['bmi'],df['bmi_corrected'])

# %%
plt.plot(df['bmi'],df['bmi_estimated'])

# %%
plt.plot(df['bmi_scaled'],df['bmi'])

# %%
df_drop = ['survey_code', 'bmi_scaled', 'bmi_scaled', 'bmi_corrected',  'bmi_estimated','occupation'] #['environmental_risk_score', 'electrolyte_level']
df = df.drop(columns=df_drop, errors="ignore")
print(df.shape[1])

# %% [markdown]
# Check dimensions, datatypes, and target distribution.

# %% [markdown]
# #### Dataset Overview

# %%
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes.value_counts())
print("\nTarget distribution:\n", df["target"].value_counts(normalize=True))

# %% [markdown]
# #### Categorical Variables

# %% [markdown]
# Exploring the categorical features frequency counts and visualize:

# %%
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    print(f"\n=== {col} ===")
    print(df[col].value_counts(dropna=False).head(10))

# %% [markdown]
# Visualization (barplots for top categories):



for col in cat_cols:
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"{col} distribution")
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# #### Numerical Variables

# %% [markdown]
# Explore distributions, central tendency, and outliers:

# %%
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
print("Numerical columns:", num_cols)

# Summary statistics
df[num_cols].describe().T

# Histograms
df[num_cols].hist(bins=30, figsize=(20,15))
plt.show()

# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap (numeric features)")
plt.show()

# %% [markdown]
# Dropping unnecessary cols

# %%
df_drop_2 = ['survey_code','environmental_risk_score', 'electrolyte_level']
df = df.drop(columns=df_drop_2, errors="ignore")
print(df.shape[1])
print(df.columns)

# %% [markdown]
# rechecking

# %%
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
print("Numerical columns:", num_cols)

# Summary statistics
df[num_cols].describe().T

# Histograms
df[num_cols].hist(bins=30, figsize=(20,15))
plt.show()

# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap (numeric features)")
plt.show()

# %% [markdown]
# ### Feature vs Target Relationships

# %% [markdown]
# #### Categorical vs Target

# %%
for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index")
    print(f"\n{col} vs target:\n", ct)

# %%


focus_table = []

for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index")
    if "diseased" in ct.columns:   # safety check
        ct["diff"] = ct["diseased"] - ct["diseased"].mean()
        focus_table.append((col, ct["diff"].abs().max()))

focus_df = pd.DataFrame(focus_table, columns=["Feature", "MaxDifference"]).sort_values("MaxDifference", ascending=False)
print(focus_df)

# %% [markdown]
# **Categories vs Target Plots**

# %%


for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(6,3))
    plt.title(f"{col} vs Target")
    plt.ylabel("Proportion")
    plt.legend(title="Target")
    plt.show()

# %% [markdown]
# Categorical features in your dataset are generally weak predictors individually (max diffs <0.0005).

# %%
# Threshold for weak categorical features
threshold = 0.0005

# Get features below threshold
drop_cols_3 = focus_df.loc[focus_df["MaxDifference"] < threshold, "Feature"].tolist()

print("Categorical columns to drop (MaxDifference < 0.0005):")
print(drop_cols_3)

# %%
df = df.drop(columns=drop_cols_3, errors="ignore")
print(df.shape[1])
print(df.columns)

# %% [markdown]
# #### Numerical vs Target (boxplots)

# %%
for col in num_cols:
    if col != "target": 
        plt.figure(figsize=(6,3))
        sns.boxplot(x="target", y=col, data=df)
        plt.title(f"{col} by target")
        plt.show()

# %% [markdown]
# * Pval intpret
# 	*	High effect size + low p-value = strong, meaningful predictor.
# 	*	Low effect size + low p-value = “significant but trivial” (large sample sizes can make even tiny differences statistically significant).
# 	*	High effect size + high p-value = might be meaningful but not enough data (rare here with 100k rows).
# 	*	Low effect size + high p-value = useless predictor (drop candidate).

# %%


# Select only numeric columns (no need to drop target since it's not numeric)
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

results = []

for col in num_cols:
    diseased = df.loc[df["target"]=="diseased", col].dropna()
    healthy  = df.loc[df["target"]=="healthy", col].dropna()
    
    # Means
    mean_d = diseased.mean()
    mean_h = healthy.mean()
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((diseased.std()**2) + (healthy.std()**2)) / 2)
    effect_size = (mean_d - mean_h) / pooled_std if pooled_std > 0 else 0
    
    # T-Test p-value
    _, pval = ttest_ind(diseased, healthy, equal_var=False, nan_policy="omit")
    
    results.append((col, mean_d, mean_h, effect_size, pval))

num_focus = pd.DataFrame(results, 
                         columns=["Feature","Mean_diseased","Mean_healthy","EffectSize","p_value"]) \
            .sort_values("EffectSize", key=np.abs, ascending=False)

print(num_focus.head(15))

# %%


top_features = num_focus.head(5)["Feature"].tolist()

for col in top_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="target", y=col, data=df)
    plt.title(f"{col} vs Target")
    plt.show()

# %%
print(df.shape[1])
print(df.columns)

# %% [markdown]
# ### Skewness Checks

# %%
num_cols = [col for col in num_cols if col in df.columns]
from scipy.stats import skew

skewness = df[num_cols].apply(lambda x: skew(x.dropna()))
print("Skewness of numeric features:\n", skewness.sort_values(ascending=False))

# %% [markdown]
# No extreme skewed variables (nothing > 1) - No urgent transformations needed

# %% [markdown]
# ### Multicollinearity - Cross-feature contradictions

# %% [markdown]
# Check if some features are redundant:

# %%


corr = df[num_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [col for col in upper.columns if any(upper[col] > 0.85)]
print("Highly correlated features:", high_corr)

# %% [markdown]
# Dropping height and weight as it highlt correlate to BMI

# %%
df_drop_5 = ['height', 'weight'] 
df = df.drop(columns=df_drop_5, errors="ignore")
print(df.shape[1])

# %%
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes.value_counts())
print("\nTarget distribution:\n", df["target"].value_counts(normalize=True))



# %% [markdown]
# ### Summary

# %%
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtype != "O" and col != "target"]
bool_cols = [col for col in df.columns if df[col].dtype == "bool"]

print("Category Cols DF", cat_cols)
print("Numerical Colms DF:", num_cols)

print("--------------")



# %% [markdown]
# ### Smoothing Nosiy Data and Outlier detection

# %%
df.describe().T

# %% [markdown]
# Outlier detection

# %%


def outlier_summary(df, z_thresh=3):
    """
    Detect outliers in numeric columns using IQR and Z-score methods.
    
    Parameters:
        df : pandas DataFrame
        z_thresh : float, threshold for Z-score (default=3)
    
    Returns:
        summary_df : DataFrame with counts and % of outliers for each method
    """
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    results = []

    for col in num_cols:
        series = df[col].dropna()

        # --- IQR Method ---
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        iqr_outliers = ((series < lower) | (series > upper)).sum()

        # --- Z-score Method ---
        z_scores = zscore(series)
        z_outliers = (np.abs(z_scores) > z_thresh).sum()

        # Save results
        results.append({
            "Feature": col,
            "IQR_outliers": iqr_outliers,
            "IQR_outlier_pct": round(100 * iqr_outliers / len(series), 2),
            "Z_outliers": z_outliers,
            "Z_outlier_pct": round(100 * z_outliers / len(series), 2),
        })

    return pd.DataFrame(results).sort_values("IQR_outlier_pct", ascending=False)


# %% [markdown]
# DF

# %%
summary = outlier_summary(df)
# print(summary)

# Filter out rows where IQR_outlier_pct is 0
non_zero_outliers = summary[summary['IQR_outlier_pct'] != 0]
# Print only Feature and IQR_outlier_pct
print(non_zero_outliers[['Feature', 'IQR_outlier_pct']].to_string(index=False))


# %% [markdown]
# Potential Noisy / Outlier Issues:
# 1.	insulin: min = -6.79 → negative values are biologically impossible.
# → Likely noise or data entry error.
# 2.	daily_steps: up to 180k → unrealistic (most people <30k/day).
# → Outlier/noise.
# 3.	calorie_intake: min = 527, max = 39k → 39,000 kcal/day is impossible.
# → Severe outliers.
# 4.	sugar_intake: negative values (–27g) → not possible.
# → Noise.
# 5.	water_intake: 0–5 liters → plausible.
# 6.	mental_health_score: 0–10 → looks like a bounded scale. Noisy min/max values okay.
# 7.	daily_supplement_dosage: –9.99 min → negative dosage is invalid.

# %% [markdown]
# Applying cleaning to df (cap only flagged columns)
# *	clean_df_full(df): for tree models (RF/XGB/LGBM).
# *	Keep all rows.
# *	Clip implausible/extreme values (domain caps + 1–99% winsorization).
# *	Return a short report of what changed.

# %%
# Non-zero outlier list
flag_df_cols = [
    "insulin","heart_rate","blood_pressure","bmi","calorie_intake","waist_size",
    "daily_steps","sugar_intake","cholesterol","glucose","work_hours","water_intake",
    "screen_time","weight","physical_activity","sleep_hours"
]

# keep only those that exist
flag_df_cols = [c for c in flag_df_cols if c in df.columns]

def iqr_clip_columns(frame: pd.DataFrame, cols, k=1.5):
    """Clip specified columns to IQR fences [Q1-k*IQR, Q3+k*IQR]."""
    frame = frame.copy()
    changed_counts = {}
    for col in cols:
        Q1, Q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - k*IQR, Q3 + k*IQR
        before = frame[col].copy()
        frame[col] = frame[col].clip(lo, hi)
        changed = int((before != frame[col]).sum())
        if changed:
            changed_counts[col] = changed
    return frame, changed_counts

df, df_changes = iqr_clip_columns(df, flag_df_cols, k=1.5)
print("Dictionary report showing how many values were clipped in each column.")
print("Capped (df):", df_changes)



# %% [markdown]
# Adding safety domain caps such as no negative insulin

# %%
DOMAIN_BOUNDS = {
    "insulin": (0, 300),
    "daily_steps": (0, 35000),
    "calorie_intake": (800, 6000),
    "sugar_intake": (0, 300),
    "heart_rate": (30, 220),
    "blood_pressure": (60, 220),
}

for col, (lo, hi) in DOMAIN_BOUNDS.items():
    if col in df.columns:
        before = df[col].copy()
        df[col] = df[col].clip(lo, hi)
        changed = int((before != df[col]).sum())
        if changed:
            df_changes[col] = df_changes.get(col, 0) + changed

print("Capped + domain (df):", df_changes)

# %%
df.shape

# %% [markdown]
# ## Data Transformation - Categorical Encoding

# %%
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtype != "O" and col != "target"]
bool_cols = [col for col in df.columns if df[col].dtype == "bool"]

print(df.shape)
print("Category Cols DF", cat_cols)
print("Numerical Colms DF:", num_cols)

print("--------------")

# %% [markdown]
# * Options:
# 	*	OneHotEncoding → expands each category into binary dummy columns. Works for both linear and tree models, but can blow up dimensionality if categories are many.
# 	*	OrdinalEncoding → assigns integers to categories. Okay for trees, risky for linear models (they interpret numbers as ordered).
# 	*	Target Encoding / Leave-One-Out → useful for high-cardinality features, but risk of leakage.

# %% [markdown]
# *	df (trees) → you can use OrdinalEncoding (trees split based on thresholds).|

# %%


TARGET_COL = "target"

# 1. Identify categorical columns (excluding target)
cat_cols = [c for c in df.columns if df[c].dtype == "O" and c != TARGET_COL]

# 2. Copy DataFrame for tree models
df_tree = df.copy()

# 3. Encode categorical features
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_tree[cat_cols] = encoder.fit_transform(df_tree[cat_cols])

# 4. Encode target with LabelEncoder
target_encoder = LabelEncoder()
df_tree[TARGET_COL] = target_encoder.fit_transform(df_tree[TARGET_COL])

# 5. Forward + reverse maps for features
label_to_code = {
    col: {label: i for i, label in enumerate(cats)}
    for col, cats in zip(cat_cols, encoder.categories_)
}
code_to_label = {
    col: {i: label for i, label in enumerate(cats)}
    for col, cats in zip(cat_cols, encoder.categories_)
}

# 6. Forward + reverse maps for target
target_label_to_code = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
target_code_to_label = dict(zip(target_encoder.transform(target_encoder.classes_), target_encoder.classes_))

# 7. Example: decode a few rows back to strings (features only)
decoded_features = pd.DataFrame(
    encoder.inverse_transform(df_tree[cat_cols].values),
    columns=cat_cols
).head()

print("Feature label -> code (example):", list(label_to_code.items())[:1])
print("Target mapping:", target_label_to_code)
print("Decoded feature preview:\n", decoded_features.head())

# %% [markdown]
# # Linear Correlation Between Variables and Target

# %%


# Select only numeric columns
numeric_df = df_tree.select_dtypes(include='number')

# Get correlation matrix
corr_matrix = numeric_df.corr()

# Correlation with target
target_corr = corr_matrix['target'].drop('target')

# Features positively correlated with unhealthy (target=1)
unhealthy_corr = target_corr.sort_values(ascending=False).head(10)

# Features positively correlated with healthy (target=0)
healthy_corr = target_corr.sort_values(ascending=True).head(10)

# Combine
combined_corr = pd.concat([
    unhealthy_corr.rename("Unhealthy (target=1)"),
    healthy_corr.rename("Healthy (target=0)")
])

# Plot side-by-side
plt.figure(figsize=(10, 6))
sns.barplot(x=combined_corr.values, y=combined_corr.index, palette="coolwarm")
plt.title("Top Features Related to Healthy (0) vs Unhealthy (1)")
plt.xlabel("Correlation with target (0=healthy, 1=unhealthy)")
plt.axvline(0, color="gray", linestyle="--")
plt.tight_layout()
plt.show()

# Print numeric values clearly
print("Top correlations with Unhealthy (1):\n", unhealthy_corr)
print("\nTop correlations with Healthy (0):\n", healthy_corr)

# %% [markdown]
# # Model Trainings and Accuracy Testing

# %% [markdown]
# ## Logistic Regression Model

# %%


X = df_tree.drop(columns=["target"])
y = df_tree["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Target distribution in train:", y_train.value_counts(normalize=True))

# %%


# Example pipeline with imputation + scaling + logistic regression
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),   # or "median", "most_frequent"
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=500, solver="liblinear"))
])

pipe.fit(X_train, y_train)

# %%


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%


log_reg = LogisticRegression(max_iter=500, solver="liblinear", class_weight=None)
log_reg.fit(X_train_scaled, y_train)

# %%


y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(log_reg, X, y, cv=cv, scoring="roc_auc")

print("Cross-val ROC-AUC:", np.mean(scores), "±", np.std(scores))

# %%


dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
proba = dummy.predict_proba(X_test)[:,1]
print("Dummy ROC-AUC:", roc_auc_score(y_test, proba))

# %%
from sklearn.metrics import roc_auc_score
aucs = {}
for col in X_train.columns:
    try:
        aucs[col] = roc_auc_score(y_train, X_train[col])
    except: 
        continue
print(sorted(aucs.items(), key=lambda x: abs(x[1]-0.5), reverse=True)[:20])

# %% [markdown]
# ## Tree based

# %%



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, classification_report,
    confusion_matrix
)

# Use df_tree_final (all numeric, target in {0,1})
TARGET_COL = "target"
X = df_tree.drop(columns=[TARGET_COL])
y = df_tree[TARGET_COL].astype(int)

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test :", y_test.value_counts(normalize=True).round(3).to_dict())

# helper: scan thresholds and pick best
def tune_threshold(y_true, y_proba, metric="f1", grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 37)
    best_thr, best_score = 0.5, -1
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden_j":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            score = tpr + tnr - 1
        else:
            raise ValueError("metric must be 'f1', 'balanced_accuracy', or 'youden_j'")
        if score > best_score:
            best_score, best_thr = score, t
    return best_thr, best_score

def evaluate_at_threshold(y_true, y_proba, thr, title="Model"):
    y_pred = (y_proba >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc  = roc_auc_score(y_true, y_proba)

    print(f"\n=== {title} (thr={thr:.3f}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=3, zero_division=0))

# %% [markdown]
# ### Random Forest

# %%


rf = RandomForestClassifier(
    n_estimators=400,
    n_jobs=-1,
    class_weight="balanced",   # make it care about the minority class
    random_state=42
)

# small validation from train for threshold tuning
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)

rf.fit(X_tr, y_tr)
val_proba = rf.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")  # or "f1"

test_proba = rf.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="RandomForest")

# %% [markdown]
# ###  HistGradientBoosting

# %%


# weight positives more (compute scale w.r.t. train)
pos = (y_train == 1).sum(); neg = (y_train == 0).sum()
hgb = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=500,
    class_weight={0: 1.0, 1: neg/max(pos,1)},   # similar to scale_pos_weight
    random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)

hgb.fit(X_tr, y_tr)
val_proba = hgb.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba = hgb.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="HistGradientBoosting")

# %% [markdown]
# ### XGBoost

# %% [markdown]
# 1.	trains XGBoost and tunes the threshold
# 2.	computes SHAP on a sample
# 3.	builds shap_importance_df = mean(|SHAP|) per feature
# 4.	selects features with mean(|SHAP|) > 0.03 (with a fallback if too few)
# 5.	retrains on just those features, re-tunes the threshold, and evaluates

# %%


def tune_threshold_for_recall0(y_true, proba1, *,
                               min_precision1=0.65,   # keep some precision on class 1
                               min_accuracy=None,     # or require min overall accuracy
                               grid=None):
    """
    Pick the threshold that maximizes recall for class 0 (specificity),
    while meeting optional constraints.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    best_thr, best_rec0 = 0.5, -1.0
    best_stats = {}
    for t in grid:
        y_pred = (proba1 >= t).astype(int)
        rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc   = accuracy_score(y_true, y_pred)
        ok = True
        if min_precision1 is not None:
            ok &= (prec1 >= min_precision1)
        if min_accuracy is not None:
            ok &= (acc >= min_accuracy)
        if ok and rec0 > best_rec0:
            best_rec0, best_thr = rec0, t
            best_stats = {"recall0": rec0, "precision1": prec1, "accuracy": acc}
    # fallback: no threshold met constraints → take highest recall0 regardless
    if best_rec0 < 0:
        for t in grid:
            y_pred = (proba1 >= t).astype(int)
            rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            if rec0 > best_rec0:
                best_rec0, best_thr = rec0, t
                best_stats = {}
    return float(best_thr), best_stats

def evaluate_with_thr(y_true, proba1, thr, title="Model"):
    y_pred = (proba1 >= thr).astype(int)
    print(f"\n=== {title} (thr={thr:.3f}) ===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

# %%


def predict_at_threshold(proba1, thr):
    return (proba1 >= thr).astype(int)

def best_thr_for_recall0(y_true, proba1, n_steps=199):
    thrs = np.linspace(0.01, 0.99, n_steps)
    best_thr, best = 0.5, -1.0
    for t in thrs:
        rec0 = recall_score(y_true, predict_at_threshold(proba1, t), pos_label=0)
        if rec0 > best:
            best_thr, best = t, rec0
    return best_thr, best

def make_sample_weight(y, boost=1.0):
    y = np.asarray(y)
    n0 = (y==0).sum(); n1 = (y==1).sum()
    if n0==0 or n1==0: return np.ones_like(y, float)
    w0 = (n1/n0) * boost        # upweight class 0
    return np.where(y==0, w0, 1.0).astype(float)

# %%


# --- 1) Train base XGB and tune threshold on validation
xgb = XGBClassifier(
    tree_method="hist",            
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)
# before: xgb.fit(X_tr, y_tr)
sw_tr = make_sample_weight(y_tr, boost=1.5)   # try 1.5–2.0
xgb.fit(X_tr, y_tr, sample_weight=sw_tr)

val_proba = xgb.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba = xgb.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="XGBoost (all features)")

# --- 2) SHAP on a sample and build shap_importance_df
n_sample = min(1000, len(X_test))        # keep it brisk; adjust if you want
X_shap = X_test.sample(n_sample, random_state=42)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap, check_additivity=False)  # (n_rows, n_features)

mean_abs = np.abs(shap_values).mean(axis=0)         # mean |SHAP| per feature
shap_importance_df = (pd.DataFrame({
    "Feature": X_shap.columns,
    "mean(|SHAP|)": mean_abs
}).sort_values("mean(|SHAP|)", ascending=False))

print("\nTop SHAP features (global mean |SHAP|):")
print(shap_importance_df.head(15).to_string(index=False))

# Optional: quick global bar
shap.summary_plot(shap_values, X_shap, plot_type="bar")

# --- 3) Select features with mean(|SHAP|) > 0.03 (fallback to top 10 if too few)
THRESH = 0.03
sel_feats = list(shap_importance_df.loc[shap_importance_df["mean(|SHAP|)"] > THRESH, "Feature"])

if len(sel_feats) < 3:  # fallback so we don't end up with too tiny a set
    sel_feats = list(shap_importance_df.head(10)["Feature"])
    print(f"\n[Fallback] Fewer than 3 features above {THRESH}; using top 10 by SHAP instead.")
else:
    print(f"\nSelected {len(sel_feats)} features with mean(|SHAP|) > {THRESH}:")
    print(sel_feats)

# --- 4) Retrain using only the selected features
X_train_sel = X_train[sel_feats].copy()
X_test_sel  = X_test[sel_feats].copy()

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=123, stratify=y_train
)

xgb_red = XGBClassifier(
    tree_method="hist",
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
# before: xgb_red.fit(X_tr2, y_tr2)
sw_tr2 = make_sample_weight(y_tr2, boost=1.5) # same boost here
xgb_red.fit(X_tr2, y_tr2, sample_weight=sw_tr2)

val_proba2 = xgb_red.predict_proba(X_val2)[:, 1]
thr_r0_2, stats2 = tune_threshold_for_recall0(
    y_val2, val_proba2,
    min_precision1=0.70,   # tweak these two knobs to your liking
    min_accuracy=0.69
)
print("Chosen thr (reduced):", thr_r0_2)
print(stats2)

test_proba2 = xgb_red.predict_proba(X_test_sel)[:, 1]
evaluate_with_thr(y_test, test_proba2, thr_r0_2, title="XGB (reduced) – recall0-optimized")

# %% [markdown]
# ###  LightGBM

# %%


lgbm = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)

lgbm.fit(X_tr, y_tr)
val_proba = lgbm.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba = lgbm.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="LightGBM")

# %% [markdown]
# ### CatBoost

# %%


cat = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",  # handles imbalance internally
    verbose=0,
    random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)

cat.fit(X_tr, y_tr)
val_proba = cat.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba = cat.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="CatBoost")

# %% [markdown]
# ### Tree Based Models Testing  - Training models Based on Top Influential Features Extraction - Other than SHAP

# %%


def metrics_at_threshold(y_true, y_proba, thr: float):
    y_pred = (y_proba >= thr).astype(int)

    # per-class precision (order: [0, 1])
    prec_per_class = precision_score(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "precision_0": float(prec_per_class[0]),
        "precision_1": float(prec_per_class[1]),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "y_pred": y_pred,
    }

# %%


def model_importance_df(model, X, y=None):
    # CatBoost native
    try:
        from catboost import CatBoostClassifier, Pool
        if isinstance(model, CatBoostClassifier):
            imp = model.get_feature_importance(Pool(X, y), type="FeatureImportance")
            return (pd.DataFrame({"Feature": X.columns, "importance": imp})
                    .sort_values("importance", ascending=False))
    except Exception:
        pass

    # Tree models with .feature_importances_
    if hasattr(model, "feature_importances_"):
        return (pd.DataFrame({"Feature": X.columns, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False))

    # Fallback: permutation importance (fast sample)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    Xs, ys = X.iloc[idx], (y.iloc[idx] if y is not None else None)
    perm = permutation_importance(model, Xs, ys, n_repeats=5, random_state=42, n_jobs=-1)
    return (pd.DataFrame({"Feature": X.columns, "importance": np.abs(perm.importances_mean)})
            .sort_values("importance", ascending=False))


def run_importance_reduction_with_metrics(
    base_model, name,
    X_train, y_train, X_test, y_test,
    top_k=12, importance_min=None,
    thr_metric="balanced_accuracy",
    print_top=12
):
    # ===== Baseline =====
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=123
    )
    model = base_model.__class__(**base_model.get_params())
    model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_val)[:, 1]
    best_thr, _ = tune_threshold(y_val, val_proba, metric=thr_metric)

    test_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} (all features) (thr={best_thr:.3f}) ===")
    evaluate_at_threshold(y_test, test_proba, thr=best_thr, title=f"{name} (all features)")
    base_metrics = metrics_at_threshold(y_test, test_proba, best_thr)

    

    # ===== Importance & selection =====
    imp_df = model_importance_df(model, X_train, y_train)
    print("\nTop features by model importance:")
    print(imp_df.head(print_top).to_string(index=False))

    if importance_min is not None:
        imp_cut = imp_df[imp_df["importance"] >= importance_min]
        sel = imp_cut.head(top_k) if len(imp_cut) > 0 else imp_df.head(top_k)
    else:
        sel = imp_df.head(top_k)

    selected_feats = list(sel["Feature"])
    print(f"\nSelected {len(selected_feats)} features:\n{selected_feats}")

    # ===== Reduced =====
    X_train_sel = X_train[selected_feats].copy()
    X_test_sel  = X_test[selected_feats].copy()

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_train_sel, y_train, test_size=0.2, stratify=y_train, random_state=123
    )
    model2 = base_model.__class__(**base_model.get_params())
    model2.fit(X_tr2, y_tr2)

    val_proba2 = model2.predict_proba(X_val2)[:, 1]
    best_thr2, best_score2 = tune_threshold(y_val2, val_proba2, metric=thr_metric)

    test_proba2 = model2.predict_proba(X_test_sel)[:, 1]
    print(f"\n=== {name} (top-{len(selected_feats)} feats) (thr={best_thr2:.3f}) ===")
    evaluate_at_threshold(y_test, test_proba2, thr=best_thr2, title=f"{name} (reduced)")
    red_metrics = metrics_at_threshold(y_test, test_proba2, best_thr2)

    return {
        "name": name,
        "baseline": base_metrics,
        "reduced": red_metrics,
        "selected_feats": selected_feats,
        "importance_table": imp_df
    }

# %%
results = []

# Example models (add/remove as you like)


xgb = XGBClassifier(
    tree_method="hist",
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
lgbm = LGBMClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, is_unbalance=True
)
rf = RandomForestClassifier(
    n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1
)
hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_bins=255, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
cat = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
                         auto_class_weights="Balanced", verbose=0, random_state=42)

for model, name in [
    (xgb, "XGBoost"),
    (lgbm, "LightGBM"),
    (rf, "RandomForest"),
    (hgb, "HistGradientBoosting"),
    (dt, "DecisionTree"),
    (cat, "CatBoost"),
]:
    res = run_importance_reduction_with_metrics(
        model, name,
        X_train, y_train, X_test, y_test,
        top_k=12,              # keep top 12
        importance_min=None,   # or set a floor e.g. 0.002
        thr_metric="balanced_accuracy"
    )
    results.append(res)

# %%


# Build a small table of metrics
rows = []
for r in results:
    rows.append({
        "model": r["name"],
        "acc_baseline": r["baseline"]["accuracy"],
        "acc_reduced":  r["reduced"]["accuracy"],
        "delta_acc":    r["reduced"]["accuracy"] - r["baseline"]["accuracy"],
        "f1_baseline":  r["baseline"]["f1"],
        "f1_reduced":   r["reduced"]["f1"],
        "delta_f1":     r["reduced"]["f1"] - r["baseline"]["f1"],
        "bal_acc_base": r["baseline"]["balanced_accuracy"],
        "bal_acc_red":  r["reduced"]["balanced_accuracy"],
        "delta_bal":    r["reduced"]["balanced_accuracy"] - r["baseline"]["balanced_accuracy"],
        "roc_base":     r["baseline"]["roc_auc"],
        "roc_red":      r["reduced"]["roc_auc"],
        "delta_roc":    r["reduced"]["roc_auc"] - r["baseline"]["roc_auc"],
    })
    
perf_df = pd.DataFrame(rows).sort_values("acc_reduced", ascending=False)
print("\nModel performance (baseline vs reduced):")
print(perf_df[["model","acc_baseline","acc_reduced","delta_acc",
               "f1_baseline","f1_reduced","delta_f1",
               "bal_acc_base","bal_acc_red","delta_bal",
               "roc_base","roc_red","delta_roc"]]
      .round(4).to_string(index=False))

# Bar plot: accuracy before/after
labels = perf_df["model"].values
x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(10,5))
b1 = ax.bar(x - w/2, perf_df["acc_baseline"].values, width=w, label="Baseline")
b2 = ax.bar(x + w/2, perf_df["acc_reduced"].values, width=w, label="Reduced (top features)")

ax.set_title("Accuracy: Baseline vs Reduced (feature-importance selection)")
ax.set_xticks(x, labels, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy")
ax.legend()

# annotate deltas above bars (reduced)
for xi, v in zip(x, perf_df["acc_reduced"].values):
    ax.text(xi + w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

for xi, v in zip(x, perf_df["acc_baseline"].values):
    ax.text(xi - w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# Quick textual summary of increases/decreases
delta_txt = (perf_df[["model","delta_acc","delta_f1","delta_bal","delta_roc"]]
             .round(4)
             .rename(columns={"delta_acc":"Δacc", "delta_f1":"Δf1", "delta_bal":"ΔbalAcc", "delta_roc":"ΔROC"}))
print("\nDeltas (reduced - baseline):")
print(delta_txt.to_string(index=False))

# %% [markdown]
# ## Model Selection

# %% [markdown]
# ### XG Boost

# %% [markdown]
# 1.	trains XGBoost and tunes the threshold
# 2.	computes SHAP on a sample
# 3.	builds shap_importance_df = mean(|SHAP|) per feature
# 4.	selects features with mean(|SHAP|) > 0.03 (with a fallback if too few)
# 5.	retrains on just those features, re-tunes the threshold, and evaluates

# %%


# --- 1) Train base XGB and tune threshold on validation
xgb = XGBClassifier(
    tree_method="hist",            
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)
# before: xgb.fit(X_tr, y_tr)
sw_tr = make_sample_weight(y_tr, boost=1.5)   # try 1.5–2.0
xgb.fit(X_tr, y_tr, sample_weight=sw_tr)

val_proba = xgb.predict_proba(X_val)[:, 1]
best_thr, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba = xgb.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba, thr=best_thr, title="XGBoost (all features)")

# --- 2) SHAP on a sample and build shap_importance_df
n_sample = min(1000, len(X_test))        # keep it brisk; adjust if you want
X_shap = X_test.sample(n_sample, random_state=42)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap, check_additivity=False)  # (n_rows, n_features)

mean_abs = np.abs(shap_values).mean(axis=0)         # mean |SHAP| per feature
shap_importance_df = (pd.DataFrame({
    "Feature": X_shap.columns,
    "mean(|SHAP|)": mean_abs
}).sort_values("mean(|SHAP|)", ascending=False))

print("\nTop SHAP features (global mean |SHAP|):")
print(shap_importance_df.head(15).to_string(index=False))

# Optional: quick global bar
shap.summary_plot(shap_values, X_shap, plot_type="bar")

# --- 3) Select features with mean(|SHAP|) > 0.03 (fallback to top 10 if too few)
THRESH = 0.03
sel_feats = list(shap_importance_df.loc[shap_importance_df["mean(|SHAP|)"] > THRESH, "Feature"])

if len(sel_feats) < 3:  # fallback so we don't end up with too tiny a set
    sel_feats = list(shap_importance_df.head(10)["Feature"])
    print(f"\n[Fallback] Fewer than 3 features above {THRESH}; using top 10 by SHAP instead.")
else:
    print(f"\nSelected {len(sel_feats)} features with mean(|SHAP|) > {THRESH}:")
    print(sel_feats)

# --- 4) Retrain using only the selected features
X_train_sel = X_train[sel_feats].copy()
X_test_sel  = X_test[sel_feats].copy()

X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=123, stratify=y_train
)

xgb_red = XGBClassifier(
    tree_method="hist",
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
# before: xgb_red.fit(X_tr2, y_tr2)
sw_tr2 = make_sample_weight(y_tr2, boost=1.5) # same boost here
xgb_red.fit(X_tr2, y_tr2, sample_weight=sw_tr2)

val_proba2 = xgb_red.predict_proba(X_val2)[:, 1]
thr_r0_2, stats2 = tune_threshold_for_recall0(
    y_val2, val_proba2,
    min_precision1=0.70,   # tweak these two knobs to your liking
    min_accuracy=0.69
)
print("Chosen thr (reduced):", thr_r0_2)
print(stats2)

test_proba2 = xgb_red.predict_proba(X_test_sel)[:, 1]
evaluate_with_thr(y_test, test_proba2, thr_r0_2, title="XGB (reduced) – recall-optimized")

# %% [markdown]
# ### Selecting model to dumbs

# %%


# Save model and selected features
joblib.dump(xgb_red, "xgb_model.pkl")
joblib.dump(sel_feats, "selected_features.pkl")
joblib.dump(thr_r0_2, "best_threshold.pkl")

# %% [markdown]
# ### Exploring the final Selected Features for Frontend

# %%
selected_feats = [
    "sugar_intake", "bmi", "cholesterol", "sleep_hours", 
    "physical_activity", "work_hours", "blood_pressure", 
    "calorie_intake", "water_intake", "daily_supplement_dosage", 
    "screen_time", "glucose", "insulin", "age", "daily_steps"
]

# %%
# Keep only selected features + target
df_filtered = df_tree[selected_feats + ["target"]]

print("Filtered dataset shape:", df_filtered.shape)
print(df_filtered.head())

# %%
df_decoded = df_tree.copy()

for col in cat_cols:
    if col in df_tree.columns:
        df_decoded[col] = df_tree[col].map(code_to_label[col])

# %%
for col in cat_cols:
    if col in code_to_label:
        print(f"\n{col} mapping:")
        print(code_to_label[col])

# %%


# Summary stats
print(df_filtered.describe().T)

# Histograms by target
for col in selected_feats:
    plt.figure(figsize=(6,3))
    sns.histplot(data=df_filtered, x=col, hue="target", kde=True, bins=30)
    plt.title(f"{col} distribution by target")
    plt.show()

# %% [markdown]
# # Finetuning Model

# %%


# ============================================================
# 1) Train base XGB on ALL features and tune threshold on a val split
# ============================================================
xgb = XGBClassifier(
    tree_method="hist",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123, stratify=y_train
)

xgb.fit(X_tr, y_tr)

val_proba = xgb.predict_proba(X_val)[:, 1]
best_thr_all, _ = tune_threshold(y_val, val_proba, metric="balanced_accuracy")

test_proba_all = xgb.predict_proba(X_test)[:, 1]
evaluate_at_threshold(y_test, test_proba_all, thr=best_thr_all, title="XGBoost (all features)")

# ============================================================
# 2) SHAP importances → feature list + tidy DataFrame you can reuse
# ============================================================
n_sample = min(1000, len(X_test))  # speed-up for SHAP
X_shap = X_test.sample(n_sample, random_state=42)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap, check_additivity=False)  # (n_rows, n_features)

shap_importance_df = (
    pd.DataFrame({
        "Feature": X_shap.columns,
        "mean(|SHAP|)": np.abs(shap_values).mean(axis=0)
    })
    .sort_values("mean(|SHAP|)", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 15 by mean(|SHAP|):")
print(shap_importance_df.head(15).to_string(index=False))

# === Selection rule (edit here) ===
THRESH = 0.03
TOP_FALLBACK = 10
sel_feats = shap_importance_df.loc[shap_importance_df["mean(|SHAP|)"] > THRESH, "Feature"].tolist()
if len(sel_feats) < 3:
    sel_feats = shap_importance_df.head(TOP_FALLBACK)["Feature"].tolist()
    print(f"[Fallback] Using top {TOP_FALLBACK} features by SHAP since <3 passed THRESH={THRESH}")

# Optional: hard-override to try a custom subset (comment out if not needed)
# sel_feats = ['bmi', 'cholesterol', 'sleep_hours', 'daily_steps', 'sugar_intake']

# Keep a tidy record
selected_features_df = pd.DataFrame({"SelectedFeature": sel_feats})
print(f"\nSelected {len(sel_feats)} features:\n{sel_feats}")

# ============================================================
# 3) Build SEPARATE DataFrames for reduced feature training & testing
# ============================================================
X_train_sel = X_train[sel_feats].copy()
X_test_sel  = X_test[sel_feats].copy()

# If you want a fresh val split only on selected features:
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_sel, y_train, test_size=0.2, random_state=123, stratify=y_train
)

# Handy containers you can import/use elsewhere:
feature_store = {
    "all_features": list(X_train.columns),
    "selected_features": sel_feats,
    "shap_importance_df": shap_importance_df
}

# ============================================================
# 4) Train the REDUCED XGBoost model (only selected features)
# ============================================================
xgb_red = XGBClassifier(
    tree_method="hist",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
)
xgb_red.fit(X_tr2, y_tr2)

val_proba2 = xgb_red.predict_proba(X_val2)[:, 1]
best_thr2, best_score2 = tune_threshold(y_val2, val_proba2, metric="balanced_accuracy")
print(f"\n[Reduced] Best threshold: {best_thr2:.3f} (bal-acc={best_score2:.4f})")

test_proba2 = xgb_red.predict_proba(X_test_sel)[:, 1]
evaluate_at_threshold(y_test, test_proba2, thr=best_thr2, title=f"XGBoost (reduced: {len(sel_feats)} feats)")

# ============================================================
# 5) (Optional) Quick fine-tuning on REDUCED features
#    → swap xgb_red with best estimator if it improves val metric
# ============================================================
param_dist = {
    "n_estimators": [200, 300, 400, 600, 800],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [1, 2, 3, 5],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.5, 1.0],
    "reg_alpha": [0, 0.001, 0.01, 0.1],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

xgb_base = XGBClassifier(
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=25,
    scoring="balanced_accuracy",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)
search.fit(X_tr2, y_tr2)

# Evaluate the tuned candidate on the same val split + threshold tuning
xgb_tuned = search.best_estimator_
val_proba_tuned = xgb_tuned.predict_proba(X_val2)[:, 1]
best_thr_tuned, best_score_tuned = tune_threshold(y_val2, val_proba_tuned, metric="balanced_accuracy")
print(f"[Tuned] val bal-acc={best_score_tuned:.4f} @ thr={best_thr_tuned:.3f}")
print("Best params:", search.best_params_)

# If tuned is better, use it going forward:
use_tuned = best_score_tuned > best_score2
final_model = xgb_tuned if use_tuned else xgb_red
final_thr = best_thr_tuned if use_tuned else best_thr2

test_proba_final = final_model.predict_proba(X_test_sel)[:, 1]
evaluate_at_threshold(y_test, test_proba_final, thr=final_thr, title="XGBoost (FINAL reduced)")

# %% [markdown]
# Compute univariate AUC for each feature (does any single feature separate 0 vs 1?).
# If all features are ≈0.5 → dataset has no predictive signal.

# %%

aucs = {}
for col in X_train.columns:
    try:
        aucs[col] = roc_auc_score(y_train, X_train[col])
    except: 
        continue
print(sorted(aucs.items(), key=lambda x: abs(x[1]-0.5), reverse=True)[:20])

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# * New features like:
# 	*	Risk composites (metabolic_risk, cardio_risk, obesity_flag)
# 	*	Lifestyle indices (sleep_efficiency, work_life_balance, stress_sleep_gap, activity_ratio)
# 	*	Likert transforms (stress_cat, mental_cat)
# 	*	Critical interactions (high_stress_low_support)
# 	*	Ratios (waist_height_ratio, sugar_ratio, water_per_weight, energy_balance)
# * A table of top 20 features ranked by univariate ROC-AUC.
# * If you see some engineered features jump to 0.55–0.65+, you now have predictive signal.

# %%
# ---------------------------
# 1. Copy original dataframe
# ---------------------------
df_eng = df_tree.copy()

# Ensure target is binary numeric
df_eng["target"] = df_eng["target"]

# ===========================
# 2. Feature Engineering
# ===========================

# --- Health Risk Scores ---
df_eng["metabolic_risk"] = (
    df_eng["bmi"] +
    df_eng["waist_size"]/100 +
    df_eng["cholesterol"]/200 +
    df_eng["glucose"]/100 +
    df_eng["blood_pressure"]/120 +
    df_eng["insulin"]/10
)

df_eng["cardio_risk"] = df_eng["blood_pressure"] * df_eng["cholesterol"] / (df_eng["heart_rate"] + 1)

df_eng["obesity_flag"] = ((df_eng["bmi"] >= 30) | (df_eng["waist_size"] > 100)).astype(int)

# --- Behavioral / Lifestyle ---
df_eng["sleep_efficiency"] = df_eng["sleep_quality"].map({
    "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4
}) * df_eng["sleep_hours"]

df_eng["work_life_balance"] = df_eng["work_hours"] / (
    df_eng["sleep_hours"] + df_eng["physical_activity"] + (df_eng["daily_steps"]/1000) + 1e-6
)

df_eng["stress_sleep_gap"] = df_eng["stress_level"] - df_eng["sleep_hours"]

df_eng["activity_ratio"] = df_eng["daily_steps"] / (df_eng["work_hours"] + 1)

# --- Likert transforms ---
df_eng["stress_cat"] = pd.cut(
    df_eng["stress_level"], bins=[-1,3,6,10], labels=["low","moderate","high"]
)
df_eng["mental_cat"] = pd.cut(
    df_eng["mental_health_score"], bins=[-1,4,7,10], labels=["poor","average","good"]
)

# Interaction: high stress with no support
df_eng["high_stress_low_support"] = (
    (df_eng["stress_level"] >= 7) & (df_eng["mental_health_support"] == "No")
).astype(int)

# --- Ratios ---
df_eng["waist_height_ratio"] = df_eng["waist_size"] / (df_eng["height"]+1e-6) if "height" in df_eng.columns else df_eng["waist_size"]/df_eng["bmi"]

df_eng["sugar_ratio"] = df_eng["sugar_intake"] / (df_eng["calorie_intake"] + 1e-6)

df_eng["water_per_weight"] = df_eng["water_intake"] / (df_eng["weight"]+1e-6) if "weight" in df_eng.columns else df_eng["water_intake"]/(df_eng["bmi"]+1e-6)

df_eng["energy_balance"] = df_eng["calorie_intake"] / (df_eng["daily_steps"] + 1)


# ===========================
# 3. Univariate AUCs
# ===========================
y = df_eng["target"]
aucs = {}

for col in df_eng.columns:
    if col == "target":
        continue
    try:
        x = df_eng[col]
        # If categorical, convert to dummy and take mean prob
        if x.dtype == "O" or str(x.dtype).startswith("category"):
            x = pd.get_dummies(x, drop_first=True)
            score = roc_auc_score(y, x.mean(axis=1))
        else:
            score = roc_auc_score(y, x)
        aucs[col] = score
    except Exception:
        continue

aucs_sorted = sorted(aucs.items(), key=lambda x: abs(x[1]-0.5), reverse=True)

print("\n=== Top 20 Features by univariate AUC ===")
for feat, auc in aucs_sorted[:20]:
    print(f"{feat:25s} AUC={auc:.3f}")

# %% [markdown]
# 1.	The dataset has almost no discriminative power in its current form.
# 	*	Even combinations like BMI + glucose + cholesterol (metabolic risk) are ≈ random.
# 	*	This suggests the target (healthy vs diseased) is not directly explained by these measurements.
# 2.	Possible issues:
# 	*	Label noise: Are the “diseased” vs “healthy” labels generated synthetically or misaligned?
# 	*	Feature leakage gap: Maybe the true predictors (e.g., diagnosis results, medications, hospital visits) aren’t in the dataset.
# 	*	Scaling/coding issues: Likert and categorical encodings might be reducing signal if not mapped correctly.
# 	*	Too balanced distributions: From your histograms, most features are centered with normal-like distributions → little separation between groups.

# %% [markdown]
# ### A dummy classifier

# %%


dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
proba = dummy.predict_proba(X_test)[:,1]
print("Dummy ROC-AUC:", roc_auc_score(y_test, proba))

# %% [markdown]
# ### Stratify features by target

# %%


for col in ["bmi", "cholesterol", "glucose", "stress_level"]:
    sns.kdeplot(data=df_eng, x=col, hue="target", common_norm=False)
    plt.title(f"{col} by target")
    plt.show()

# %% [markdown]
# ## Adding Polynomial Interactions

# %% [markdown]
# * Why this helps
# 	*	Instead of looking at each variable in isolation, this will automatically generate feature × feature interactions (e.g., bmi * glucose, stress * sleep, cholesterol * activity).
# 	*	Logistic regression (linear on these expanded features) can now pick up nonlinear decision boundaries.
# 	*	If disease risk emerges only when two conditions coincide (like high BMI + high cholesterol), you’ll finally see separation.

# %%


# --------------------------
# 1. Select numeric features
# --------------------------
num_cols = ['age', 'bmi', 'waist_size', 'blood_pressure', 
            'heart_rate', 'cholesterol', 'glucose', 'insulin', 
            'sleep_hours', 'work_hours', 'physical_activity', 
            'daily_steps', 'calorie_intake', 'sugar_intake', 
            'water_intake', 'screen_time', 'stress_level', 
            'mental_health_score', 'meals_per_day', 
            'daily_supplement_dosage']

X = df_eng[num_cols]
y = df_eng["target"]

# --------------------------
# 2. Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 3. Pipeline with poly features
# --------------------------
pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

# --------------------------
# 4. Evaluate
# --------------------------
proba = pipe.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
print("ROC-AUC with polynomial interactions:", auc)

# %% [markdown]
# 1.	builds pairwise interaction features for numeric columns,
# 2.	one-hot encodes categoricals,
# 3.	selects the Top-K features by mutual information (fast + model-agnostic), and
# 4.	trains XGBoost on the reduced matrix, picking a threshold that maximizes balanced accuracy.

# %%
# ============================================================
# 0) Imports
# ============================================================




# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def threshold_at_max_balacc(y_true, proba, grid_size=500):
    thr_grid = np.linspace(0.01, 0.99, grid_size)
    best_thr, best_balacc = 0.5, -1.0
    for t in thr_grid:
        y_hat = (proba >= t).astype(int)
        bal = balanced_accuracy_score(y_true, y_hat)
        if bal > best_balacc:
            best_balacc, best_thr = bal, t
    return float(best_thr), float(best_balacc)

def evaluate_all(y_true, proba, thr, title="Model"):
    y_pred = (proba >= thr).astype(int)
    acc  = (y_pred == y_true).mean()
    prec = ((y_pred==1) & (y_true==1)).sum() / max((y_pred==1).sum(), 1)
    rec  = ((y_pred==1) & (y_true==1)).sum() / max((y_true==1).sum(), 1)
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    roc  = roc_auc_score(y_true, proba)
    pr   = average_precision_score(y_true, proba)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n=== {title} (thr={thr:.3f}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"PR-AUC   : {pr:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

# ============================================================
# 1) Inputs: column lists (from your message)
# ============================================================
cat_cols = ['sleep_quality','smoking_level','mental_health_support',
            'education_level','job_type','diet_type','exercise_type','device_usage',
            'healthcare_access', 'caffeine_intake']

num_cols = ['age', 'bmi', 'waist_size', 'blood_pressure', 
            'heart_rate', 'cholesterol', 'glucose', 'insulin', 
            'sleep_hours', 'work_hours', 'physical_activity', 
            'daily_steps', 'calorie_intake', 'sugar_intake', 
            'water_intake', 'screen_time', 'stress_level', 
            'mental_health_score', 'meals_per_day', 
            'daily_supplement_dosage']
# ============================================================
# 2) Prepare data
# ============================================================
df = df_tree.copy()

TARGET_COL = "target"
X = df_tree.drop(columns=[TARGET_COL])
y = df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3) Preprocess:
#    - OneHot for categoricals
#    - Numeric passthrough
#    - Numeric pairwise interactions (degree=2, interaction_only=True)
#    Output is a single sparse matrix with feature names
# ============================================================
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Two separate transforms on the SAME numeric block:
pre = ColumnTransformer(
    transformers=[
        ("cat_ohe", ohe, cat_cols),
        ("num_pass", "passthrough", num_cols),
        ("num_inter", poly, num_cols),
    ],
    remainder='drop',
    sparse_threshold=1.0,  # keep sparse for memory
)

# Fit on train only
pre.fit(X_train)

# Transform to sparse matrices
Xtr = pre.transform(X_train)   # scipy.sparse
Xte = pre.transform(X_test)

# Build feature names to align with mutual_info selection
cat_names = pre.named_transformers_['cat_ohe'].get_feature_names_out(cat_cols)
num_pass_names = np.array(num_cols)
# names for interactions:
poly_names_raw = pre.named_transformers_['num_inter'].get_feature_names_out(num_cols)
# Remove any duplicates with passthrough (shouldn’t happen with interaction_only=True)
feat_names = np.concatenate([cat_names, num_pass_names, poly_names_raw])

assert Xtr.shape[1] == len(feat_names)

# ============================================================
# 4) Mutual Information feature selection (fast, model-agnostic)
#    - Compute MI on a stratified sample to avoid dense conversion blow-up
# ============================================================
rng = np.random.RandomState(42)
sample_idx = rng.choice(Xtr.shape[0], size=min(30000, Xtr.shape[0]), replace=False)
Xtr_s = Xtr[sample_idx]
ytr_s = y_train.iloc[sample_idx].values

# mutual_info_classif requires dense; convert ONLY for the sample.
# (If memory is tight, you can convert by blocks; below is simple & usually fine for 30k rows)
Xtr_s_dense = Xtr_s.toarray()

# Discrete mask: OneHot columns are discrete; numeric & interactions are continuous
n_cat = len(cat_names)
n_num = len(num_pass_names)
n_int = len(poly_names_raw)
discrete_mask = np.zeros(Xtr_s_dense.shape[1], dtype=bool)
discrete_mask[:n_cat] = True  # one-hots

mi = mutual_info_classif(Xtr_s_dense, ytr_s, discrete_features=discrete_mask, random_state=42)

# Pick Top-K by MI
K = 150  # adjust (100–300 is typical); you can print a curve later and refine
topk_idx = np.argpartition(mi, -K)[-K:]
topk_idx = topk_idx[np.argsort(mi[topk_idx])[::-1]]  # sort descending
topk_names = feat_names[topk_idx]

print(f"\n[MI] Selected Top-{K} features.")
print(topk_names[:25])  # peek first 25

# Reduce train/test to selected columns
Xtr_sel = Xtr[:, topk_idx]
Xte_sel = Xte[:, topk_idx]

# ============================================================
# 5) Train XGBoost on the MI-reduced matrix
#    - handle class imbalance via scale_pos_weight
# ============================================================
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
spw = neg / max(pos, 1)

xgb = XGBClassifier(
    tree_method="hist",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.01,
    reg_lambda=1.5,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    scale_pos_weight=spw,
)

xgb.fit(Xtr_sel, y_train)

# ============================================================
# 6) Threshold by MAX balanced accuracy on a small holdout from train
# ============================================================
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    Xtr_sel, y_train, test_size=0.2, random_state=123, stratify=y_train
)
xgb.fit(X_tr2, y_tr2)




# %%
val_proba = xgb.predict_proba(X_val2)[:, 1]
best_thr2, best_score2 = tune_threshold(y_val2, val_proba, metric="balanced_accuracy")
print(f"\nBest thr on SHAP-filtered set: {best_thr2:.3f} (bal-acc={best_score2:.4f})")

test_proba = xgb.predict_proba(Xte_sel)[:, 1]
evaluate_at_threshold(y_test, test_proba2, thr=best_thr2, title=f"XGBoost (SHAP>{THRESH} features)")

# %%
val_proba = xgb.predict_proba(X_val2)[:, 1]
thr_bal, bal_val = threshold_at_max_balacc(y_val2, val_proba)
print(f"\n[Threshold] Val best balanced accuracy = {bal_val:.4f} @ thr = {thr_bal:.3f}")

test_proba = xgb.predict_proba(Xte_sel)[:, 1]
evaluate_with_thr(y_test, test_proba2, thr_r0_2, title=f"XGB + Poly Interactions + MI Top-{K}")

# %%



