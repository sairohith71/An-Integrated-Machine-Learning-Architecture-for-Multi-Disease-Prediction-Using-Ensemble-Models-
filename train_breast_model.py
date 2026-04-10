import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("datasets/breast_cancer/data.csv")

# Remove unnecessary columns
if "id" in data.columns:
    data = data.drop("id", axis=1)

if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)

# Convert diagnosis column (M=1, B=0)
data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})

# Features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

print("Feature count:", len(X.columns))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("breast_model.sav", "wb"))

print("Breast Cancer Model Saved Successfully!")