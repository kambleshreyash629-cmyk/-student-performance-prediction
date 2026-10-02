import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Features for our first model
features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

# Input features
X = df[features]

# Target
y = df["G3"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())