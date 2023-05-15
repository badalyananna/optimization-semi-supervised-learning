import time
import numpy as np
from common import calc_grad_full, calc_obj_func

def GM(y_l, W_l, W_u, alpha, max_iter, eps, calc_of=False):
    """Algorithms that implements gradient method.
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled
    W_u - similarity matrix for unlabelled
    alpha - fixed step size
    max-iter - maximum number of iteration
    eps - tolerance
    calc_of - boolean variable, if True the objective functions is calucated at each iteration
    """
    start_time = time.time()
    grad_stat = []
    obj_fun_stat = []
    time_stat = [0]
    
    n = len(W_u)
    y_pred = np.zeros(n)

        
    obj_fun_val = calc_obj_func(y_pred, y_l, W_l, W_u)
    obj_fun_stat.append(obj_fun_val)
    print(f"Initial loss: {obj_fun_val}")
    
    for i in range(max_iter):
        gradient = calc_grad_full(y_pred, y_l, W_l, W_u)
        direction = - gradient

        y_pred += alpha * direction

        stop_cond = abs(direction @ gradient) / n
        
        if calc_of:
            obj_fun_val = calc_obj_func(y_pred, y_l, W_l, W_u)
            obj_fun_stat.append(obj_fun_val)
            print(f"Iteration {i+1}: obj fun {obj_fun_val}, gradient norm {stop_cond}")
        else:
            print(f"Iteration {i+1}: gradient norm {stop_cond}")
            
        iter_time = time.time()
        time_stat.append(iter_time - start_time)
        grad_stat.append(stop_cond)
        if stop_cond < eps:
            obj_fun = calc_obj_func(y_pred, y_l, W_l, W_u)
            obj_fun_stat.append(obj_fun)
            print(f"""
            Stopping condition satisfied, obj fun: {obj_fun}, 
            Gradient norm: {stop_cond}",
            Total CPU time: {iter_time-start_time}""")
            break
    return y_pred, grad_stat, time_stat, obj_fun_stat