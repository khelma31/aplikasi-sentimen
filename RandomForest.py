import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForestCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            
            indices = np.random.choice(len(y), len(y), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(criterion='gini', max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        return [Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(tree_predictions.shape[1])]