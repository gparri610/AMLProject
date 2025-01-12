from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForest():

    def __init__(self, n_estimators, max_depth) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion="squared_error", max_depth=max_depth)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        self.model = self.model.fit(X_train, y_train)
        training_loss = np.sum((self.model.predict(X_train) - y_train)**2)
        # Sum of Squared Error
        return training_loss

    def __call__(self, X):
        return self.model.predict(X)