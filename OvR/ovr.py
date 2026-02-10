import numpy as np

class OneVsRestClassifier:
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_params = {}

        if hasattr(estimator, "eta"):
            self.estimator_params["eta"] = getattr(estimator, "eta")
        if hasattr(estimator, "n_iter"):
            self.estimator_params["n_iter"] = getattr(estimator, "n_iter")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.estimators_ = []
        arr2d = []

        for i in self.classes_:
            tmp = np.where(y == i, 1, 0)
            arr2d.append(tmp)

        self.y_binz = np.column_stack(arr2d)

        for i in range(self.n_classes):
            estimator_i = self.estimator.__class__(**self.estimator_params)
            estimator_i.fit(X, self.y_binz[:, i])
            self.estimators_.append(estimator_i)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        pscores = np.zeros((n_samples, self.n_classes))

        for i, model in enumerate(self.estimators_):
            scores[:, i] = model.net_input(X)
            pscores[:, i] = model.predict(X)

        class_indices = np.argmax(scores, axis=1)
        return (self.classes_[class_indices], pscores)
