from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


log_model = LogisticRegression(max_iter=200)
knn_model = KNeighborsClassifier()
tree_model = DecisionTreeClassifier()


log_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)


log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
tree_pred = tree_model.predict(X_test)


log_acc = accuracy_score(y_test, log_pred)
knn_acc = accuracy_score(y_test, knn_pred)
tree_acc = accuracy_score(y_test, tree_pred)


print("\nModel Accuracy Comparison")

print(f"Logistic Regression: {log_acc:.2f}")
print(f"KNN: {knn_acc:.2f}")
print(f"Decision Tree: {tree_acc:.2f}")