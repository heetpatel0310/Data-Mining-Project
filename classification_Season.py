import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("flight_data.csv")  # Replace with the correct file name

# Encode categorical features
categorical_cols = ["WEEKDAY", "DEPARTURE PART OF THE DAY", "ARRIVAL PART OF THE DAY", "AIRLINE"]
data_encoded = data.copy()

for col in categorical_cols:
    data_encoded[col] = data[col].astype("category").cat.codes

# Correct the SEASON encoding to ensure proper index-to-season mapping
season_mapping = {"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}
data["SEASON"] = data["SEASON"].map(season_mapping)

# Ensure encoded target and mapping are consistent
data_encoded["SEASON"] = data["SEASON"]

# Select target and features
target = "SEASON"
features = ["DISTANCE", "PRICE", "REMAINING_SEATS", "WEEKDAY", "DEPARTURE PART OF THE DAY", "ARRIVAL PART OF THE DAY"]

# Split data into train and test sets
X = data_encoded[features]
y = data_encoded[target]  # Use corrected SEASON mapping
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(season_mapping.keys())))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=season_mapping.keys(), yticklabels=season_mapping.keys())
for arrow in plt.gcf().findobj(plt.matplotlib.patches.FancyArrowPatch):
    arrow.set_mutation_scale(6)  # Reduce arrow size (default is higher)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("decision_tree_confusion_matrix.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(100, 40))
plot_tree(
    clf,
    feature_names=features,
    class_names=list(season_mapping.keys()),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Visualization for Target: SEASON")
plt.savefig("decision_tree_visualization_season.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Export tree rules
tree_rules = export_text(clf, feature_names=features)
print("\nDecision Tree Rules:")
print(tree_rules)
