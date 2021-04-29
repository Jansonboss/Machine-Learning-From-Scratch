import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:, j] = (X[:, j] - u) / s


def MSE(X, y, B, lmbda):
    """
    X, y, beta: B are all numpy array
    X: (n, p) Matrix
    Y: (n, 1) vector
    B: (p, 1) vector
    """
    error = y - np.dot(X, B)
    # eTe => error vector dot itsef = square of norm
    # notice that mse is a quadratic equation -> convex guarenteed
    sse = np.linalg.norm(error) ** 2  # sum of squared errors
    num_observations = X.shape[0]  # num_observations = n

    # return mean square of error
    return sse / num_observations
    # For optimization, we don’t care about scaling MSE by 1/n
    # (though we have to be careful to adjust
    # the learning rate), giving our “loss” function:


def loss_gradient(X, y, B, lmbda):
    """
    This function implement the gradient of loss function: MSE
    By using caculas method. You can also use numerical way
    to approach the gradient of loss function
    https://cs231n.github.io/optimization-1/

    X, y, beta: B are all numpy array
    X: (n, p) Matrix
    Y: (p, 1) vector
    B: (p, 1) vector
    """
    error = y - X.dot(B)  # (n, 1)
    gradientOfLossFucntion = -np.dot(X.T, error)  # (p, n) @ (n,1)
    return gradientOfLossFucntion  # (p, 1)


def loss_ridge(X, y, B, lmbda):
    """
    X, y, beta: B are all numpy array
    X: (n, p) Matrix
    Y: (p, 1) vector
    B: (p, 1) vector
    """
    error = y - X.dot(B)  # error:dim = (p,1)
    loss = np.dot(error, error) + lmbda * B.dot(B)
    return loss


def loss_gradient_ridge(X, y, B, lmbda):

    error = y - X.dot(B)  # (n,1) - (n,p) @ (p, 1) -> (n, 1)
    gradientOfLossFucntion = -X.T.dot(error) + lmbda * B  # (n,p).T @ (n, 1) + (p,1)
    return gradientOfLossFucntion  # (p, 1) contains partial derotive vector


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, B, lmbda):
    """
    return negative of loglikelihood function
    """
    dot_product = X.dot(B)
    # conditional prob -> P(Y=1|X=x) = sigmoid(theta.T*X)
    p = sigmoid(dot_product)
    log_likelihood = np.sum(y * np.log(p) + (1-y)*np.log(1-p))

    return -log_likelihood


def log_likelihood_gradient(X, y, B, lmbda):

    return -X.T.dot((y - sigmoid(X.dot(B.T))))


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    pass


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    pass


def minimize(X, y, loss_gradient, eta=0.00001,
            lmbda=0.0, max_iter=1000, addB0=True, precision=1e-9):

    """Performing AdaGradient Descent to minimize

    Args:
        X ([type]): [description]
        y ([type]): [description]
        loss_gradient ([type]): [description]
        eta (float, optional): [description]. Defaults to 0.00001.
            eta will set learing rate (Step Size)
        lmbda (float, optional): [description]. Defaults to 0.0.
        max_iter (int, optional): [description]. Defaults to 1000.
        addB0 (bool, optional): [description]. Defaults to True.

        precision (float, optional): [description]. Defaults to 0.00000001.
    """
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")

    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    # add column of 1s to X
    # https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
    if addB0:
        # .c_ goes for add columns, .r_: add row
        X = np.c_[np.ones(n), X]
        p += 1  # add extra dim for beta_0

    # initialize beta vector as ranndom vector, make between [-1,1)
    B = np.random.random_sample(size=(p, 1)) * 2 - 1

    # store history of beta vector: useful for adagradient
    # eps -> prevent division by 0
    prev_B, eps = B, 1e-5
    history = np.zeros((p, 1))

    index = 0
    while index < max_iter:
        # gradientOFLoss: numpy array contain gradient at x point (p, 1)
        grad_loss = loss_gradient(X, y, B, lmbda)
        history += grad_loss ** 2
        # Important: do not assign it like below otherwise the scale vector
        # will be really small and later on gradient will explode and never
        # converge scale_vector = 1 / np.sqrt(history + eps)

        if np.linalg.norm(grad_loss) < precision:
            print("gradient decent finished, solution gets converged")
            return B
        # elementwise muplication
        B = B - eta / np.sqrt(history + eps) * grad_loss
        index += 1

    # if number of iteration get access predefined level
    print("iteration over max level, not converged stop iterating")
    return B


class LinearRegression621:  # REQUIRED
    """
    usage:
        regr = LinearRegression621()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
    """
    def __init__(self, eta=0.00001, lmbda=0.0, max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(
            X, y, loss_gradient, self.eta,
            self.lmbda, self.max_iter)


class LogisticRegression621:  # REQUIRED
    "Use the above class as a guide."

    def __init__(self, eta, lmbda=0.0, max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def fit(self, X, y):
        self.B = minimize(X, y, log_likelihood, log_likelihood_gradient,
            self.eta, self.lmbda, self.max_iter)

    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])

        predicted_prob = sigmoid(X.dot(self.B))  # might be B not B.T

        return np.c_[1-predicted_prob, predicted_prob]

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        p = self.predict_proba(X)
        true_false = p[:, 0] < p[:, 1]
        # compare first col and second col
        return true_false.astype(int)


class RidgeRegression621:  # REQUIRED
    "Use the above classes as a guide."
    """
    usage:
        regr = LinearRegression621()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
    """
    def __init__(self, eta=0.00001, lmbda=0.0, max_iter=1000, addBO=False):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.addBO = addBO

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(
            X, y, loss_gradient_ridge, self.eta,
            self.lmbda, self.max_iter, self.addBO)

        B0 = np.array(np.mean(y))
        self.B = np.vstack([B0, self.B])


# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    pass