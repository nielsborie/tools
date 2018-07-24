from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.cross_validation import StratifiedKFold

class Ensemble_stack(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(y, n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred


#ensemble method: model averaging
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
        return np.mean(predictions, axis=1)


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
