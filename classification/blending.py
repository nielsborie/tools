import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

# --- Custom class for stacking estimator's prediction in classification problem
class Blending(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_folds = 5
        self.verbose = True
        self.shuffle = False

    def fit(self, X, y):
        n_classes = len(set(y))
        self.output = np.zeros((X.shape[0], n_classes))
        skf = StratifiedKFold(y, self.n_folds)
        self.D = {}
        for i,(tra, tst) in enumerate(skf):
            self.D[i] = self.estimator.fit(X[tra], y[tra])
            
            #self.output[tst,:] = self.estimator.predict_proba(X[tst])
        #self.estimator.fit(X, self.output)
        return self

    def transform(self, X,y=None):
        X_transformed = np.zeros((X.shape[0], 1))
        for key,val in self.D.items():
            self.D[key].predict(X)
            X_transformed=np.c_[X_transformed,self.D[key].predict(X)]
        return np.delete(X_transformed, 0, 1)
