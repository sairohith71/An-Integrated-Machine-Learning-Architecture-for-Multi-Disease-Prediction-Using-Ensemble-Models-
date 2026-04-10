import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("datasets/lung_cancer/lung_cancer.csv")

# remove spaces
data.columns = data.columns.str.strip()

# convert gender
data["GENDER"] = data["GENDER"].map({"M":1, "F":0})

# convert YES/NO → 1/2
for col in data.columns:
    if data[col].dtype == object:
        data[col] = data[col].map({"YES":1, "NO":2})

# features
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("models/lung_model.sav", "wb"))

print("Model trained successfully")