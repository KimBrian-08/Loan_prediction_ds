class ModelFactory:
    def create_model(self, model_type, **params):
        if model_type == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(**params)
        # Add other models as needed
        raise ValueError(f'Unknown model type: {model_type}')

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)