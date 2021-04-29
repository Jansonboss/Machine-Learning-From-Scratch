from contextlib import redirect_stderr
import numpy as np
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:- 1]
    return X, y

def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """Given X, a array y and num_iter return y_mean and trees 
   
    Input: X, y, num_iter
           max_depth
           nu (is the shinkage)
    Outputs:y_mean, array of trees from DecisionTreeRegression
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y) # intialize the f0(x)
    fm = y_mean.repeat(len(X))       # m = 0 -> f0
    for m in range(0,num_iter):
        residual = y - fm 
        Tm = DecisionTreeRegressor(max_depth=max_depth)
        Tm.fit(X=X, y=residual)
        trees.append(Tm)
        # learning rate should be applied on iutput of stump
        fm = fm + nu*Tm.predict(X)
    return y_mean, trees  

def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """Given X, trees, y_mean predict y_hat
    """
    yhat = np.array(y_mean.repeat(len(X)))
    for t in trees:
        yhat = yhat + nu*t.predict(X)

    return yhat

