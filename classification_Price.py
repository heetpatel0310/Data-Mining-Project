import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("flight_data.csv")

# Ensure the required columns are available
required_columns = [
    "DISTANCE",
    "REMAINING_SEATS",
    "SEASON",
    "AIRLINE",
    "PRICE",
    "DEPARTURE PART OF THE DAY",
    "ARRIVAL PART OF THE DAY",
    "WEEKDAY",
]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {set(required_columns) - set(data.columns)}")

# Create PRICE_CATEGORY by binning the PRICE column
price_bins = [0, 200, 400, 600, float('inf')]
price_labels = ["Low", "Medium", "High", "Premium"]
data["PRICE_CATEGORY"] = pd.cut(data["PRICE"], bins=price_bins, labels=price_labels, right=False)

# Encode categorical variables
data["SEASON"] = data["SEASON"].astype("category").cat.codes
data["AIRLINE"] = data["AIRLINE"].astype("category").cat.codes
data["DEPARTURE PART OF THE DAY"] = data["DEPARTURE PART OF THE DAY"].astype("category").cat.codes
data["ARRIVAL PART OF THE DAY"] = data["ARRIVAL PART OF THE DAY"].astype("category").cat.codes
data["WEEKDAY"] = data["WEEKDAY"].astype("category").cat.codes

# Prepare features and target
X = data[
    [
        "DISTANCE",
        "REMAINING_SEATS",
        "SEASON",
        "AIRLINE",
        "DEPARTURE PART OF THE DAY",
        "ARRIVAL PART OF THE DAY",
        "WEEKDAY",
    ]
]
y = data["PRICE_CATEGORY"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_classifier.fit(X_train, y_train)

# Generate textual representation of the tree and save to file
tree_text = export_text(dt_classifier, feature_names=X_train.columns.tolist())
print("Decision Tree Textual Representation:\n")
print(tree_text)

with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_text)

# Plot the decision tree and save as PDF
plt.figure(figsize=(120, 60))
plot_tree(
    dt_classifier,
    feature_names=X_train.columns,
    class_names=dt_classifier.classes_,
    filled=True,
    rounded=True,
    max_depth=6  # Adjust for complexity
)


# Adjust arrow sizes
for arrow in plt.gcf().findobj(plt.matplotlib.patches.FancyArrowPatch):
    arrow.set_mutation_scale(6)  # Reduce arrow size (default is higher)

plt.title("Comprehensive Decision Tree Visualization")
plt.savefig("decision_tree_visualization.pdf", format="pdf", bbox_inches="tight")
plt.show()
