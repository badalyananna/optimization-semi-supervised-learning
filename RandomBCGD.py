import time
import numpy as np
from common import calc_grad_y_j, calc_obj_func

def rand_BCGD(y_l, W_l, W_u, L_i, max_cycle, eps, nesterov_sampling=False, calc_of=False, alpha_min=0, random_seed=None):
    """Implements the randomized BCGD method to minimize the objective function.
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled
    W_u - similarity matrix for unlabelled
    L_i - Lipschitz constant for each coordinate
    alpha_min - sets the minimum value for the stepsize
    max_cycle - maximum number of cycles
    eps - tolerance
    nesterov_samling - if true, samples the coordinates according to the Nesterov's rule,
        choosing the coordinates with the largest L_i more often
    calc_of - boolean variable, if True the objective functions is calucated after each cycle
    """
    start_time = time.time()
    np.random.seed(random_seed)
    
    time_stat = [0] 
    grad_stat = []
    obj_fun_stat = []
    
    n = len(W_u)
    y_pred = np.zeros(n)
    alpha = 1 / L_i
    if nesterov_sampling:
        p = L_i / L_i.sum()
        
    obj_fun = calc_obj_func(y_pred, y_l, W_l, W_u)
    print(f"Initial loss: {obj_fun}")

    for it in range(max_cycle):
        stop_cond = 0
        for _ in range(n):
            # choose random coordinate
            if nesterov_sampling:
                jk = np.random.choice(n, p=p)
            else:
                jk = np.random.randint(n)
            y_j = y_pred[jk]

            # calculate the gradient
            grad = calc_grad_y_j(jk, y_pred, y_l, W_l, W_u)
            direction = - grad

            # update predictions
            alpha_k = max(alpha[jk], alpha_min)
            u = alpha_k * direction
            y_j += u
            y_pred[jk] = y_j
        
            stop_cond += abs(grad*direction)
            
        stop_cond /= n # take the average of gradient and direction products
        grad_stat.append(stop_cond)
        iter_time = time.time()
        time_stat.append(iter_time-start_time)
        if stop_cond < eps:
            obj_fun = calc_obj_func(y_pred, y_l, W_l, W_u)
            obj_fun_stat.append(obj_fun)
            print(f"""
            Stopping condition satisfied, obj fun: {obj_fun}, 
            Gradient norm: {stop_cond}",
            Total CPU time: {iter_time-start_time}""")
            break
        
        if calc_of:
            obj_fun_val = calc_obj_func(y_pred, y_l, W_l, W_u)
            obj_fun_stat.append(obj_fun_val)
            print(f"Cycle {it+1}: obj fun {obj_fun_val}, gradient norm {stop_cond}")
        else:
            print(f"Cycle {it+1}: gradient norm {stop_cond}")
     
    return y_pred, grad_stat, time_stat, obj_fun_stat