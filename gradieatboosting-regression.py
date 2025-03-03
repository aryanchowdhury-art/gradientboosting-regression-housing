import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset (Replace with actual dataset path or method)
df = pd.read_csv("housing.csv")

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# Convert categorical feature ('ocean_proximity') into numerical using one-hot encoding
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Define features and target (Replace 'median_house_value' with your actual target column name)
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and Gradient Boosting Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor())
])

# Define hyperparameter grid
param_grid = {
    "gbr__n_estimators": [100, 200, 300],   # Number of boosting stages
    "gbr__learning_rate": [0.01, 0.05, 0.1],  # Step size shrinkage
    "gbr__max_depth": [3, 5, 7],  # Maximum depth of trees
    "gbr__min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "gbr__min_samples_leaf": [1, 3, 5]  # Minimum samples per leaf
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
