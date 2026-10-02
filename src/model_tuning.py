import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------
# 1. Load dataset
# ---------------------------------------
df = pd.read_csv("data/student-mat.csv", sep=";")


# ---------------------------------------
# 2. Features and target
# ---------------------------------------
features = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

X = df[features]
y = df["G3"]


# ---------------------------------------
# 3. Categorical columns
# ---------------------------------------
categorical_features = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]


# ---------------------------------------
# 4. Preprocessing
# ---------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ---------------------------------------
# 5. Create pipeline
# ---------------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42
            )
        )
    ]
)


# ---------------------------------------
# 6. Keep test data separate
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------
# 7. Parameters to test
# ---------------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# ---------------------------------------
# 8. Grid Search with 5-fold CV
# ---------------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)


# ---------------------------------------
# 9. Train and search
# ---------------------------------------
grid_search.fit(X_train, y_train)


# ---------------------------------------
# 10. Best configuration
# ---------------------------------------
print("===== MODEL TUNING RESULTS =====")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation MAE:")
print(-grid_search.best_score_)


# ---------------------------------------
# 11. Final evaluation on untouched test set
# ---------------------------------------
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n===== FINAL TEST RESULTS =====")

print("Test MAE:", mae)
print("Test MSE:", mse)
print("Test R2:", r2)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nActual Values:")
print(y_test.values[:10])
