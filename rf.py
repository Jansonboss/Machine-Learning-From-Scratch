import numpy as np
import math
from sklearn.metrics import r2_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split,
        self.lchild = lchild
        self.rchild = rchild
    
    def predict(self, X_test):
        if X_test[self.col] <= self.split:
            return self.lchild.predict(X_test)
        return self.rchild.predict(X_test)
    
    def leaf(self, X_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        return self.predict(X_test) # trigger leafnode prediction (return leaf)


class LeafNode:
    def __init__(self, y, prediction):
        self.y = y
        self.n = len(y)
        self.prediction = prediction
    
    def predict(self, X_test):
        # when reaching the leaf return the leaf it self
        # include x_test just wanna keep the interface constant as DecisionNode.predict
        if self.prediction: return self
            

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        
    def _RFBestSplit(self, X, y, loss, max_features):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        best = -1, -1, loss(y) # col, split, loss
        feature_subset_idx = np.random.choice(X.shape[1], # selecting column feature index subset
                                                # do not use floor, not able to pass the unit test r^2 low for some re
                                                math.ceil(max_features * X.shape[1]), replace=True)
        for col in feature_subset_idx:
            X_col = X[:, col]
            # randomly choose 11 point as split spots and find the bast one with lowest gini or std
            split_candidate  = np.random.choice(X_col, 15)
            for split in split_candidate:
                yl, yr = y[X_col <= split], y[X_col > split] #<= is important
                yl_size, yr_size = len(yl), len(yr)
                if yl_size <= self.min_samples_leaf or yr_size <= self.min_samples_leaf:
                    continue
                l = (yl_size * loss(yl) + yr_size * loss(yr)) / (yl_size + yr_size)
                if l == 0: return best[0], best[1] # return col, and split_candidate
                if l < best[-1]: best = col, split, l
        return best[0], best[1]

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.trees_ = []
        for i in range(self.n_estimators):
            inbagX, inbagy, oobX, ooby = bootStrapping(X=X, y=y)
            tree = self._fit(X=inbagX, y=inbagy)
            self.trees_.append(tree)

    def _fit(self, X, y):
        self.nclass = np.unique(y)
        if len(X) <= self.min_samples_leaf or len(self.nclass)==1:
            return LeafNode(y=y, prediction=self.prediction)
        col, split = self._RFBestSplit(X, y, loss=self.loss, max_features=self.max_features)
        if col == -1: return LeafNode(y=y, prediction=self.prediction)
        lchild = self._fit(X = X[X[:, col]<=split], y=y[X[:, col]<=split])
        rchild = self._fit(X = X[X[:, col]>split], y=y[X[:, col]>split])
        return DecisionNode(col=col, split=split,lchild = lchild, rchild =rchild)
    

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.prediction = "regress"
        self.oob_score = oob_score
        self.loss = np.std

    def predict(self, X_test):
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test. return numpy array.
        """    
        y_pred = np.zeros(len(X_test))
        for row_idx, row in enumerate(X_test):  #for each row in X-test
            leaves_size, ypred_sum = 0, 0
            for tree in self.trees_:
                myleaf = tree.leaf(row)
                ypred_sum += np.mean(myleaf.y) * myleaf.n
                leaves_size += myleaf.n
            y_pred[row_idx] = ypred_sum / leaves_size
        return y_pred
        
    def score(self, X_test, y_test):
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_true=y_test, y_pred=self.predict(X_test))


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.prediction = "classify"
        self.oob_score = oob_score
        self.loss = gini

    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for counter_idx in range(len(X_test)):
            count_dict = {}
            for tree in self.trees_:
                myleaf = tree.leaf(X_test[counter_idx, :]) # get leaf for single x_test 
                class_indices, count = np.unique(myleaf.y, return_counts=True)
                for class_idx, freq in zip(class_indices, count):
                    count_dict[class_idx] = count_dict.get(class_idx, 0) + freq
                # getting the class with the highest freq
                y_pred[counter_idx] = max(count_dict, key=count_dict.get)     
        return y_pred
        
    def score(self, X_test, y_test):
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return r2_score(y_true=y_test, y_pred=self.predict(X_test))

def bootStrapping(X, y):
    """bootstrapping sample with replacement
    And Return the bootstrapped sample size = orginal data size

    Args:
        X (2D numpy array): [n*p matrix]
        y (1D numpy array): [array]
    """ 
    n = len(X)
    mask = np.ones(n, dtype=bool)
    inbag_idx = np.random.randint(0, n, size=n)
    mask[inbag_idx] = False
    inBagX, inBagy = X[inbag_idx], y[inbag_idx] # in-bag-sample
    outofBagX, outofBagy = X[mask], y[mask] # out-of-bag sample
    return inBagX, inBagy, outofBagX, outofBagy

def gini(y):
    """
    Compute gini impurity from y vector of class values (from k unique values).
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y) # p: probablity ditribution (numpy array)
    return 1 - np.sum( p**2 )
