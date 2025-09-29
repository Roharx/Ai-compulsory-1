import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------

# Importing dataset:
# --------------------------------------------------------------------
dataset = pd.read_csv('housing.csv')

target_col = dataset.columns[-2]
feature_cols = dataset.columns.drop(target_col)

X = dataset.loc[:, feature_cols].values
y = dataset.iloc[:, -2].values

# print(X)
# print('--------------------')
# print(y)

# Taking care of missing data (if any):
# --------------------------------------------------------------------

# Everything is numeric except the last column
num_idx = np.arange(X.shape[1] - 1)

x_imputer = SimpleImputer(strategy='median')
X[:, num_idx] = x_imputer.fit_transform(X[:, num_idx])

# Encoding categorical data:
# --------------------------------------------------------------------

# cat_col = dataset.columns[-1]
# unique_vals = dataset[cat_col].unique()
# print("Unique categories:", unique_vals)
# print("Number of categories:", dataset[cat_col].nunique())

# get the last col
cat_idx = len(feature_cols) - 1

# oneHotEncode so we have numbers instead of strings
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [cat_idx])
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))
# print(X)

# Feature scaling:
# --------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print(X_train)
# print('--------------------------------')
# print(X_test)
# print('--------------------------------')
# print(y_train)
# print('--------------------------------')
# print(y_test)

sc = StandardScaler()

# Encoded categories don't need to be scaled
oneHotEncodedColumnNumber = len(ct.named_transformers_['encoder'].categories_[0])

X_train[:, oneHotEncodedColumnNumber:] = sc.fit_transform(X_train[:, oneHotEncodedColumnNumber:])
X_test[:,  oneHotEncodedColumnNumber:] = sc.transform(X_test[:,  oneHotEncodedColumnNumber:])

# print(X_train[0])
# print('--------------------------------')
# print(X_test[0])

# --------------------------------------------------------------------
# EDA (exploratory data analysis) - skippable but I need to see what the data looks like
# --------------------------------------------------------------------

# 1) Target distribution
# plt.figure(figsize=(6,4))
# dataset['median_house_value'].hist(bins=50)
# plt.xlabel('Median House Value (USD)')
# plt.ylabel('Count')
# plt.title('Target Distribution')
# plt.tight_layout()
# plt.savefig('eda_target_hist.png', dpi=150)
#
# # 2) Ocean proximity counts (categorical overview)
# plt.figure(figsize=(6,4))
# dataset['ocean_proximity'].value_counts().plot(kind='bar')
# plt.xlabel('Ocean Proximity')
# plt.ylabel('Count')
# plt.title('Category Counts: ocean_proximity')
# plt.tight_layout()
# plt.savefig('eda_ocean_counts.png', dpi=150)
#
# # 3) Correlation with target (numeric only)
# corr = dataset.select_dtypes(include=[np.number]).corr(numeric_only=True)['median_house_value'].sort_values(ascending=False)
# print("\nTop correlations with median_house_value:\n", corr.head(10))
# print("\nBottom correlations with median_house_value:\n", corr.tail(10))
#
# plt.figure(figsize=(7,5))
# corr.drop('median_house_value').plot(kind='bar')
# plt.ylabel('Correlation with median_house_value')
# plt.title('Numeric Feature Correlations')
# plt.tight_layout()
# plt.savefig('eda_correlations.png', dpi=150)
#
# # 4) Key scatterplots vs target (quick relationship check)
# plt.figure(figsize=(6,4))
# plt.scatter(dataset['median_income'], dataset['median_house_value'], alpha=0.3, s=10)
# plt.xlabel('Median Income')
# plt.ylabel('Median House Value')
# plt.title('Median Income vs House Value')
# plt.tight_layout()
# plt.savefig('eda_income_vs_value.png', dpi=150)
#
# plt.figure(figsize=(6,4))
# plt.scatter(dataset['housing_median_age'], dataset['median_house_value'], alpha=0.3, s=10)
# plt.xlabel('Housing Median Age')
# plt.ylabel('Median House Value')
# plt.title('Age vs House Value')
# plt.tight_layout()
# plt.savefig('eda_age_vs_value.png', dpi=150)
#
# # 5) Simple geospatial view (downsample to keep it light)
# sample = dataset.sample(n=min(5000, len(dataset)), random_state=42)
# plt.figure(figsize=(6,5))
# sc = plt.scatter(sample['longitude'], sample['latitude'],
#                  c=sample['median_house_value'], s=8, alpha=0.6)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Geography colored by House Value (sampled)')
# plt.colorbar(sc, label='Median House Value')
# plt.tight_layout()
# plt.savefig('eda_geo_value.png', dpi=150)

# --------------------------------------------------------------------
# Baseline vs. MLP modeling and evaluation
# --------------------------------------------------------------------


# 1) Baseline: Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)
lin_mae = mean_absolute_error(y_test, y_pred_lin)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
lin_r2 = r2_score(y_test, y_pred_lin)
print("\n=== Baseline: Linear Regression ===")
print(f"MAE : {lin_mae:,.0f}")
print(f"RMSE: {lin_rmse:,.0f}")
print(f"R²  : {lin_r2:.4f}")

# 2) MLP Regressor + compact grid (fast but meaningful)
mlp = MLPRegressor(
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=10,
    tol=1e-3,
    validation_fraction=0.1,
    random_state=42
)

param_grid = {
    "hidden_layer_sizes": [(64,), (64, 64)],
    "activation": ["relu"],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [1e-3],
    "batch_size": [128, 256],
}

grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for MLP…")
grid.fit(X_train, y_train)
print("\nBest hyperparameters:", grid.best_params_)
best_cv_rmse = (-grid.best_score_) ** 0.5
print(f"Best CV RMSE: {best_cv_rmse:,.0f}")

best_mlp = grid.best_estimator_

# 3) Final evaluation on held-out test set
y_pred = best_mlp.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Test Set Performance: MLP Regressor ===")
print(f"MAE : {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²  : {r2:.4f}")

# 4) Quick diagnostics: residuals vs predicted

residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals,
            c=np.abs(residuals), cmap="coolwarm",
            alpha=0.5, s=20)
plt.axhline(0, ls="--", color="black")
plt.colorbar(label="|Residual|")
plt.xlabel("Predicted house value")
plt.ylabel("Residual (True − Predicted)")
plt.title("Residuals vs Predicted — coloured by absolute error")
plt.tight_layout()
plt.savefig("residuals_coloured.png", dpi=150)