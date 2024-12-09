from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Load dataset
data = pd.read_csv("flight_data.csv")


# Define a function to generate holiday ranges (± few days)
def generate_holiday_range(base_holidays, days_range=7):
    holidays = []
    for holiday in base_holidays:
        date = datetime.strptime(holiday, "%Y-%m-%d")
        for i in range(-days_range, days_range + 1):
            holidays.append((date + timedelta(days=i)).strftime("%Y-%m-%d"))
    return holidays


# Define US holiday dates (2019-2023)
base_holiday_dates = [
    # 2019
    "2019-01-01",
    "2019-01-21",
    "2019-02-18",
    "2019-05-27",
    "2019-07-04",
    "2019-09-02",
    "2019-10-14",
    "2019-11-11",
    "2019-11-28",
    "2019-12-25",
    # 2020
    "2020-01-01",
    "2020-01-20",
    "2020-02-17",
    "2020-05-25",
    "2020-07-04",
    "2020-09-07",
    "2020-10-12",
    "2020-11-11",
    "2020-11-26",
    "2020-12-25",
    # 2021
    "2021-01-01",
    "2021-01-18",
    "2021-02-15",
    "2021-05-31",
    "2021-07-04",
    "2021-09-06",
    "2021-10-11",
    "2021-11-11",
    "2021-11-25",
    "2021-12-25",
    # 2022
    "2022-01-01",
    "2022-01-17",
    "2022-02-21",
    "2022-05-30",
    "2022-07-04",
    "2022-09-05",
    "2022-10-10",
    "2022-11-11",
    "2022-11-24",
    "2022-12-25",
    # 2023
    "2023-01-01",
    "2023-01-16",
    "2023-02-20",
    "2023-05-29",
    "2023-07-04",
    "2023-09-04",
    "2023-10-09",
    "2023-11-11",
    "2023-11-23",
    "2023-12-25",
]

# Generate extended holiday range
holiday_dates_extended = generate_holiday_range(base_holiday_dates, days_range=5)

# Standardize FL_DATE format to match holiday dates
data["FL_DATE"] = pd.to_datetime(data["FL_DATE"]).dt.strftime("%Y-%m-%d")

# Create a HOLIDAY column
data["HOLIDAY"] = data["FL_DATE"].isin(holiday_dates_extended).astype(int)

# Encode categorical variables
categorical_cols = [
    "AIRLINE",
    "AIRLINE_CODE",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY",
    "WEEKDAY",
    "DEPARTURE PART OF THE DAY",
    "ARRIVAL PART OF THE DAY",
    "SEASON",
]
data_encoded = data.copy()
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data_encoded[col] = label_encoders[col].fit_transform(data[col].astype(str))

# Prepare features and target
X = data_encoded.drop(
    [
        "HOLIDAY",
        "FL_DATE",
        "AIRLINE",
        "AIRLINE_CODE",
        "ORIGIN",
        "ORIGIN_CITY",
        "DEST",
        "DEST_CITY",
    ],
    axis=1,
)
y = data_encoded["HOLIDAY"]

# Check HOLIDAY distribution
print("HOLIDAY column value counts:")
print(data_encoded["HOLIDAY"].value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Classifier
clf_holiday = DecisionTreeClassifier(max_depth=6, random_state=42)
clf_holiday.fit(X_train, y_train)

# Evaluate model
y_pred_holiday = clf_holiday.predict(X_test)
print("\nClassification Report for HOLIDAY Classification:")
print(classification_report(y_test, y_pred_holiday))

# Cross-validation
cross_val_scores = cross_val_score(clf_holiday, X, y, cv=5)
print("\nCross-Validation Scores:", cross_val_scores)

# Visualize Decision Tree
plt.figure(figsize=(100, 30))
plot_tree(
    clf_holiday,
    feature_names=X_train.columns,
    class_names=["Non-Holiday", "Holiday"],
    filled=True,
    rounded=True,
)
plt.title("Decision Tree Visualization for HOLIDAY Classification")
plt.savefig("decision_tree_holiday_extended.pdf")
plt.show()

# Export decision tree rules
tree_rules_holiday = export_text(clf_holiday, feature_names=list(X_train.columns))
print("\nDecision Tree Rules:\n", tree_rules_holiday)

# Save tree rules to file
with open("holiday_decision_tree_rules.txt", "w") as f:
    f.write(tree_rules_holiday)

print(
    "\nHoliday classification completed. Decision tree rules saved to 'holiday_decision_tree_rules.txt'."
)
