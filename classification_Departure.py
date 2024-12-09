import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("flight_data.csv")  # Replace with your dataset path

# Preprocessing
label_encoders = {}
categorical_columns = ['AIRLINE', 'ORIGIN', 'DEST', 'WEEKDAY', 'SEASON', 'DEPARTURE PART OF THE DAY', 'ARRIVAL PART OF THE DAY']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

features = ['DISTANCE', 'PRICE', 'REMAINING_SEATS', 'ARRIVAL PART OF THE DAY']
target = 'DEPARTURE PART OF THE DAY'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Plot the Decision Tree
plt.figure(figsize=(110, 40))
tree_plot = plot_tree(
    clf,
    feature_names=features,
    class_names=[str(cat) for cat in label_encoders['DEPARTURE PART OF THE DAY'].classes_],
    filled=True,
    rounded=True,
    fontsize=12
)

# Adjust arrow sizes
for arrow in plt.gcf().findobj(plt.matplotlib.patches.FancyArrowPatch):
    arrow.set_mutation_scale(6)  # Reduce arrow size (default is higher)

# Add title and save the figure as a PDF
plt.title("Decision Tree Visualization for DEPARTURE PART OF THE DAY")
plt.savefig("decision_tree_departure_part_of_day.pdf", format="pdf", bbox_inches="tight")
plt.show()
