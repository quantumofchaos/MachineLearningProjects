from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

dataset = load_iris()

X = dataset.data[:,[2,3]]
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train: {acc_train}, Test: {acc_test}')

plt.interactive(False)
plt.title("Decision Tree con Iris")

plot_decision_regions(X_test, y_test, clf=model, legend=2)
plt.show()
