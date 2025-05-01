import pandas as pd

# Load data
df = pd.read_csv("heart.csv")

# Peek at data
print(df.head())

# Structure and data types
print(df.info())

# Summary stats
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Look at target variable distribution
print(df['target'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluate
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize
dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=False)
graph

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
importances = rf.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=features.index)
plt.title("Feature Importances")
plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

