import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "traffic_accidents_100.csv"
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Overview:\n", df.head())

# Data preprocessing
label_encoders = {}
for column in ["Weather", "Vehicle_Type", "Road_Surface"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Mapping Accident Severity to numerical values
severity_map = {"Minor": 0, "Severe": 1, "Fatal": 2}
df["Accident_Severity"] = df["Accident_Severity"].map(severity_map)

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x="Accident_Severity", data=df, palette="coolwarm")
plt.title("Accident Severity Distribution")
plt.show()

# Splitting data for prediction modeling
X = df[["Weather", "Vehicle_Type", "Road_Surface"]]
y = df["Accident_Severity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Feature importance analysis
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.plot(kind="bar", title="Feature Importance in Predicting Severity")
plt.show()
