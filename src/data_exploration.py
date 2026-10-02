import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/student-mat.csv", sep=";")

features = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G3"
]

correlation = df[features].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")

# Save the graph
plt.savefig("screenshots/correlation.png", dpi=300, bbox_inches="tight")

plt.show()