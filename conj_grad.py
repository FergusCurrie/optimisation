import numpy as np

# Example 
A = np.array([[3, 2],[2, 6]])
b = np.array([2, -8 ])
# graphing this?
# graphing quadratic form
# graphing contours of quadratic form
# graphing gradients of quadratic form 

# steepest descent for solving psd equations:

x_i = np.array([0,0])
def steepest_descent_update(x_i):
    # the update 
    r_i = b - A @ x_i
    alpha_i = np.dot(r_i.T, r_i) / np.dot(r_i.T, A @ r_i)
    x_j = x_i + alpha_i * r_i
    return x_j

def steepest_descent(x_start):
    history = [x_start]
    x_i = x_start
    for _ in range(10): # stopping criterion?
        x_j = steepest_descent_update(x_i)
        history.append(x_j)
        x_i = x_j

# jacobi splitting / jacobi iterations 
# x is stationary under repeated transofrmation,
# the transformation only efects the error part of x_i and not the correct.
# x* = optimal. e(i) = error i 
# jacobi convergence on gaurenteed 
def jacobi_method(x_start):
    D = A[np.diag(len(A))] # diagonal of A
    E = A - D # off diagonal of A
    inv_D = np.linalg.inv(D)
    B = -inv_D @ E
    z = inv_D @ b
    assert(A == D + E)

    history = [x_start]
    x_i = x_start
    for _ in range(10): # stopping criterion?
        x_j= B @ x_i + z
        history.append(x_j)
        x_i = x_j

def method_of_orthogonal_directions():
    """
    n steps to finish. 
    requires knowing the optimal solution. 

    i think error doesnt need to be split out 
    """
    # pick set of orthogonal search directions d_0, d_1... d_n
    # easiest choice for d is coordinate axis. 
    d_0 = np.array([1,0])
    d_1 = np.array([0,1])

    # we know the start and finish
    x_0 = np.array([-3, -3])
    x_opt = np.array([2 ,-2]) # is conventoin x,y or y,x 
    e = x_0 - x_opt 
    e_0 = (e @ d_0) / (np.linalg.norm(d_0, ord=2)) * d_0 # projection of error on to d_0
    e_1 = (e @ d_1) / (np.linalg.norm(d_1, ord=2)) * d_1 # projection of error on to d_1


    # in each search direction take the perfect step (based on orthogonality to error)
    alpha_0 = np.dot(d_0.T, e_0) / np.dot(d_0.T, d_0)
    alpha_1 = np.dot(d_1.T, e_1) / np.dot(d_1.T, d_1)
    
    x_1 = x_0 + alpha_0 @ d_0 
    x_2 = x_1 + alpha_1 @ d_1 

    print(f"x_start = {x_0}, x_finish = {x_2}")


def method_of_conjugate_directoins():
    """

    O(n^3)... same as gaussian elim? yup/

    - A-orthogonality
    - gram-schmidt conjugation
    - 
    """
    x_start = np.array([-3, -3])
    history = [x_start]
    x = x_start

    n = len(b)

    # construct coordinate axes
    u = []
    for i in range(n):
        u_i = np.zeros(len(n))
        u_i[i] = 1
        u.append(u_i) 

    # construct A-orthoganal directions 
    d = []
    for i in range(n):
        # the first member is just the first coordinate axes 
        if i == 0:
            d.append(u[i])
            continue 

        # conjugate gram_schmidt 
        # d_i = u_i, then remove the directions of previous (gram-schmidt)
        d_i = u[i]
        for k in range(i):
            beta_ik = -np.dot(u[i].T, A @ d[k]) / (np.dot(d[k].T, A @ d[k]))
            d_i += beta_ik * d[k] 

        d.append(d_i)

    # step through the directions 
    for i in range(n):
        residual_i = A @ x - b 
        alpha_i = np.dot(d[i].T, residual_i) / (d[i].T, A @ d[i])
        x_j = x_i + alpha_i * d[i] 
        history.append(x_j)


        
def conjugate_gradients(x_start):
    '''
    Essentially method of conjugate directions with u set to the residuals 

    CG is so impoirtant because both the space and time complexity per iteration reduce from O(n^2) to O(m),
    where m is the number of nonzero entries of A. 

    convention for variables in here is :
    r(i) = r, x(i)= x, ...
    r(i+1) = r_j, x(i+1) = x_j, ...
    '''
    history = [x_start]
    n = len(b)

    # Initalise starting values 
    x_0 = x_start 
    r_0 = b - A @ x_0
    d_0 = r_0

    # Set startin values to current
    x = x_0 
    r = r_0
    d = d_0

    for _ in range(n): 
        # Calculate this iteration 
        alpha = np.dot(r.T, r) / np.dot(d.T, A @ d)
        x_j = x + alpha * A @ d
        r_j = r - alpha * A @ d
        beta = (r_j.T, r_j) / (r.T, r)
        d_j = r_j + beta * d

        # Update parameters 
        x = x_j
        r = r_j
        d = d_j

        # update history 
        history.append(x)
        



