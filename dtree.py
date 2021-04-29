import numpy as np
from scipy.stats import mode
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self , X_test):
        # Make decicion based upon x_test[col] and split
        # predict on root node, and call leafNode.predict
        if X_test[self.col] <= self.split:
            return self.lchild.predict(X_test)
        return self.rchild.predict(X_test)

class LeafNode:
    def __init__(self, y, prediction):
        """
        Since we have prediction, we know it is leafnode; Create leaf node from y values
        prediction is mean(y) or mode(y)
        """
        self.y = y
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        if self.prediction == "regress":
            return np.mean(self.y)
        return mode(self.y)[0][0]


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """

        # Base case: reach the smallest sized node => create a leaf node
        if len(X) <= self.min_samples_leaf or len(np.unique(X))==1:
            return self.create_leaf(y=y) # return predction leaf

        col, split = self._bestsplit(X=X, y=y, loss=self.loss) # column_index, and split point
        if col == -1: # no best split found, return the whole y as leadNode(prediction)
            return self.create_leaf(y=y)

        # create left subtree and right subtree recursively
        lchild, rchild = self.fit_(X=X[X[:,col]<=split], y=y[X[:,col]<=split]), self.fit_(X=X[X[:,col]>split], y=y[X[:,col]>split])
        return DecisionNode(col, split, lchild, rchild) # decision node


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification! -> call predict on the root Node
        """
        preds = np.empty(X_test.shape[0])
        for i in range(X_test.shape[0]):
            preds[i] = self.root.predict(X_test[i])

        return preds


    def _bestsplit(self, X, y, loss):
        """
        find the best spot to split the data so that global loss is minimized
        for each features col: col_x in X, randomly select 11 unique points from col_x
        as split point (skip the split sample num in some leaves is too small), find the
        best split

        return col seletced and split
        """
        best = (-1, -1, loss(y)) ## initialize the best comb
        for col in range(X.shape[1]): # loop through col index of X
            col_x = X[:, col] # get column data
            candidate = np.random.choice(col_x, 11)
            for split in candidate:
                ly, ry = y[col_x <= split], y[col_x > split]
                ly_size, ry_size, y_size = len(ly), len(ry), len(y)

                # find feature space larger than predefined minimum sample cardinality
                if ly_size < self.min_samples_leaf or ry_size < self.min_samples_leaf:
                    continue # jump outof the for loop, avoid overfiiting

                l = (ly_size * loss(ly) + ry_size * loss(ry)) / y_size # weight average of loss
                if l ==0: return col, split
                if l < best[2]: best = (col, split, l) # update the res
        return best[0], best[1]


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_true=y_test, y_pred=self.predict(X_test=X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y=y, prediction="regress")


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        return accuracy_score(y_true=y_test, y_pred=self.predict(X_test=X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y=y, prediction='classify')


def gini(y):
    """
    Compute gini impurity from y vector of class values (from k unique values).
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y) # p: probablity ditribution (numpy array)
    return 1 - np.sum( p**2 )

