import time
import numpy as np
from common import calc_grad_y_j, calc_grad_full, calc_obj_func, update_gradient

def GS_BCGD(y_l, W_l, W_u, L_i, max_cycle, eps, calc_of=False, alpha_min=0):
    """Implements the BCGD method with Gauss Southwell rule to minimize the objective function.
    y_l - labeled variables
    W_l - similarity matrix for labelled and unlabelled
    W_u - similarity matrix for unlabelled
    L_i - Lipschitz constant for each coordinate
    alpha_min - sets the minimum value for the stepsize
    max_cycle - maximum number of cycles, 1 cycle = n iterations
    eps - tolerance
    calc_of - boolean variable, if True the objective functions is calucated after n iternations
    """
    start_time = time.time()
    
    time_stat = [0] 
    grad_stat = []
    obj_fun_stat = []
    
    n = len(W_u)
    y_pred = np.zeros(n)
    alpha = 1 / L_i
        
    obj_fun = calc_obj_func(y_pred, y_l, W_l, W_u)
    print(f"Initial loss: {obj_fun}")

    grad = calc_grad_full(y_pred, y_l, W_l, W_u)
    
    for it in range(max_iter*n):
        ik = np.argmax(np.abs(grad))
        grad_y_j = grad[ik]
        direction = grad_y_j * (-1)

        # update predictions
        alpha_k = max(alpha[ik], alpha_min)
        update = alpha_k * grad_y_j
        y_pred[ik] = y_pred[ik] - update

        # update the gradient based on new y_i
        grad = update_gradient(ik, update, grad, W_u)
        grad_y_j_new = calc_grad_y_j(ik, y_pred, y_l, W_l, W_u)
        grad[ik] = grad_y_j_new
                
        stop_cond = abs(grad_y_j*direction)
        grad_stat.append(stop_cond)
        
        if stop_cond < eps:
            time_stat.append(iter_time-start_time)
            obj_fun = calc_obj_func(y_pred, y_l, W_l, W_u)
            obj_fun_stat.append(obj_fun)
            print(f"""
            Stopping condition satisfied, obj fun: {obj_fun}, 
            Gradient norm: {stop_cond}",
            Total CPU time: {iter_time-start_time}""")
            break
        elif (it % n == 0) and (it!=0):
            if calc_of:
                obj_fun_val = calc_obj_func(y_pred, y_l, W_l, W_u)
                obj_fun_stat.append(obj_fun_val)
                print(f"Cycle {it//n}: obj fun {obj_fun_val}, gradient norm {stop_cond}")
            else:
                print(f"Cycle {it//n}: gradient norm {stop_cond}")
                    
        iter_time = time.time()
        time_stat.append(iter_time-start_time)
        
    return y_pred, grad_stat, time_stat, obj_fun_stat