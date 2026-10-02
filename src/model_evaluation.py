import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")


# Features
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


# Categorical columns
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


# Preprocessing
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


# Linear Regression pipeline
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


# Random Forest pipeline
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# 5-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Linear Regression evaluation
linear_mae = -cross_val_score(
    linear_model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

linear_r2 = cross_val_score(
    linear_model,
    X,
    y,
    cv=cv,
    scoring="r2"
)


# Random Forest evaluation
rf_mae = -cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

rf_r2 = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="r2"
)


# Display results
print("===== 5-FOLD CROSS VALIDATION =====")

print("\nLinear Regression")
print("MAE for each fold:", linear_mae)
print("Average MAE:", linear_mae.mean())
print("R2 for each fold:", linear_r2)
print("Average R2:", linear_r2.mean())

print("\nRandom Forest")
print("MAE for each fold:", rf_mae)
print("Average MAE:", rf_mae.mean())
print("R2 for each fold:", rf_r2)
print("Average R2:", rf_r2.mean())