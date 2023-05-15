import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content

def plot_data(X,Y):
    plt.scatter(X.T[0], X.T[1], c=Y)
    plt.show()

def eigval_hessian(W_l, W_u):
    """Returns the largest eigenvalue of the Hessian matrix.
    """
    n = len(W_u)
    hes = W_u * (-2)
    diag_values = (W_l.sum(axis=0) + W_u.sum(axis=0)) * 2
    np.fill_diagonal(hes, diag_values)
    max_eig = LA.eigh(hes, eigvals_only=True, subset_by_index=[n-1, n-1])[0]
    return max_eig

def calc_obj_func(y, y_l, W_l, W_u):
    """Calculates the value of the objective function.
    y - unlabelled predicted variables
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled
    W_u - similarity matrix for unlabelled
    """
    nl = len(y_l) # number of labeled points
    nu = len(y) # number of unlabeled points
    part1 = np.outer(np.ones(nl), y) - np.outer(y_l, np.ones(nu))
    part1 = np.multiply(part1, part1)
    part1 = np.multiply(W_l, part1)

    part2 = np.outer(np.ones(nu), y) - np.outer(y, np.ones(nu))
    part2 = np.multiply(part2, part2)
    part2 = np.multiply(W_u, part2)

    return np.sum(part1) + (0.5 * np.sum(part2))

def calc_grad_full(y, y_l, W_l, W_u):
    """Calculated the gradient with respect to y
    Outputs a vector of dimensions equal to y 
    y - unlabelled predicted variables
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled
    W_u - similarity matrix for unlabelled
    """
    nl = len(y_l) # number of labeled points
    nu = len(y) # number of unlabeled points
    part1 = np.outer(np.ones(nl), y) - np.outer(y_l, np.ones(nu))
    part1 = np.multiply(W_l, part1)
    part1 = np.matmul(part1.T, np.ones(nl))

    part2 = np.outer(np.ones(nu), y) - np.outer(y, np.ones(nu))
    part2 = np.multiply(W_u, part2)
    part2 = np.matmul(part2.T, np.ones(nu))
    return 2 * (part1 + part2)

def calc_grad_y_j(j, y, y_l, W_l, W_u):
    """Calculated the gradient with respec to y_j
    j - the coordinate with respect to which the gradient is calculated
    y - unlabelled predicted variables
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled points
    W_u - similarity matrix for unlabelled points
    """
    W_l_j = W_l[:,j]
    W_u_j = W_u[:,j]

    part1 = np.dot(W_l_j, y[j]-y_l)
    part2 = np.dot(W_u_j, y[j] - y)
    return 2 * (part1 + part2)

def update_gradient(ik, update, old_grad, W_u):
    """Updates the value of the gradient given the change in one coordinate
    ik - the coordinate that changed
    update - the value of the change
    old_grad - old value of the gradient
    W_u -similarity matrix for unlabelled points
    """
    return old_grad + 2 * W_u[ik] * update

def update_obj_function(jk, y_pred, y_l, W_l, W_u, u):
    """Updates the value of the obbjective function given the change in one coordinate
    jk - the coordinate that changed
    u - update, the value of the change
    """
    diff = (2 * (y_pred[jk] - y_l) - u) @ W_l[:, jk] 
    diff -= 0.5 * (2 * (y_pred - y_pred[jk]) + u) @ W_u[jk]
    diff += 0.5 * (2 * (y_pred[jk] - y_pred) - u) @ W_u[:,jk]
    return u * diff 