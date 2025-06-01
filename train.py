import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

if len(sys.argv) < 2:
    print("Usage: python train.py <input_data_file.csv or .xlsx>")
    sys.exit(1)

file_path = sys.argv[1]

# Load file based on extension
if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
    df = pd.read_excel(file_path)
else:
    print("Unsupported file type. Use .csv or .xlsx")
    sys.exit(1)

# Features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(clf, "model.joblib")

# Save metrics
with open("metrics.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"accuracy,{accuracy:.4f}\n")

print(f"Model trained and saved with accuracy: {accuracy:.4f}")
