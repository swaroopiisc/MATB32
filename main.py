#import os
import pandas as pd
import numpy as np
from scipy.optimize import fmin
from numpy.linalg import inv
#from numpy.linalg import pinv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_as_pandas_df_dat1 = pd.read_csv("/../../../../C:/Downloads/regression_1.dat", delimiter=",")
data_as_pandas_df_dat2 = pd.read_csv("../../../../../C:/Downloads/regression_2.dat", delimiter=",")

x_vals = data_as_pandas_df_dat1["x"]
y_vals = data_as_pandas_df_dat1["y"] 

def ycap_minus_y(args):
    a, b = args
    y_cap = a * x_vals + b
    error = np.sum((y_cap - y_vals)**2)
    return error

# assuming b_start = 1, then a_start = (y-1)/x This needs 27 iterations
# b_start = 1 
# a_start = (y_vals[0] - b_start)/x_vals[0]

# assuming a_start = 1, then b_start = y-x This needs 28 iterations
a_start = 1
b_start = y_vals[0] - x_vals[0]
intial_guess = [a_start, b_start]

optimal_guess = fmin(ycap_minus_y, intial_guess)


# we need to find two unknowns, a and b, from the theory
# The A matrix should be mxn, where n is the dimension of x_vector
# as we need to solve for 2 unknowns, n = 2
# we can rewrite y_cap = x * a + b as
# y_cap = [x, 1] * [a
#                   b]
# as we have an array of values for y and x, our m value will be the same as the array size
n = 2 #2 unknowns
m = len(x_vals) #can also use y_vals
A_mat = np.zeros([m,n], dtype = 'float')
A_mat[:,0] = x_vals
A_mat[:,1] = [1 for p in range(m)]

# normal eqn are A^t * A * x = A^t * y
# => x = (A^t * A)^-1 * (A^t * y)
calc_guess = inv((A_mat.T)@A_mat)@(A_mat.T)@y_vals
# 1.1 a)
print(f"Using scipy.optimize.fmin")
print(f"a = {optimal_guess[0]} | b = {optimal_guess[1]}")
# 1.1 b)
print(f"Using normal eqns and numpy")
print(f"a = {calc_guess[0]} | b = {calc_guess[1]}")
#diff = calc_guess - optimal_guess
#print(diff)

# 1.2
y_cap_vals = optimal_guess[0] * x_vals + optimal_guess[1] # Can also use : A_mat @ optimal_guess

plt.plot(x_vals, y_vals, label="given", linestyle="dashed")
plt.plot(x_vals, y_cap_vals, label="calc")
plt.xlabel("x-values")
plt.ylabel("f(x)")
plt.title("Given values vs Linear regression values")
plt.legend()
plt.grid()

plt.show()


a_vals_half1 = np.random.uniform(0.9725, optimal_guess[0], size=50)
a_vals_half2 = np.random.uniform(optimal_guess[0], 0.9825, size=50)
a_vals = np.concatenate((a_vals_half1, a_vals_half2))
b_vals_half1 = np.random.uniform(1.0743, optimal_guess[1], size=50)
b_vals_half2 = np.random.uniform(optimal_guess[1], 1.0843, size=50)
b_vals = np.concatenate((b_vals_half1, b_vals_half2))

x_chosen = 1  #chose this randomly, shouldnt affect what we are trying to do
y_opt_a_b = optimal_guess[0] * x_chosen + optimal_guess[1]

def err_fn(a,b):
    y_rand_a_b = a * x_chosen + b
    return abs(y_rand_a_b - y_opt_a_b)

err_for_opt = err_fn(optimal_guess[0], optimal_guess[1])

X, Y = np.meshgrid(a_vals, b_vals)
Z = err_fn(X,Y)
figure = plt.figure(figsize=(8, 8))
ax = figure.add_subplot(111, projection='3d')
error_surface = ax.plot_surface(X, Y, Z, cmap='magma')
cbar = figure.colorbar(error_surface, ax=ax, shrink=0.6, aspect=10, pad=0.1)
cbar.set_label("Intensity (Z-values)")
ax.scatter(optimal_guess[0], optimal_guess[1], err_for_opt, color='green', s=200, marker='s', label="Marked Point")
ax.view_init(elev=15, azim = 75)
ax.set_xlabel("a values")
ax.set_ylabel("b values")
ax.set_zlabel("error = ||y_cap - y||")
ax.set_title("surface plot of error")

#print(len(a_vals))
#print(len(b_vals))


# 1.4
# y_cap = P * y, where y_cap = Ax
# We know from normal equations
# x = (A^t * A)^-1 * A^t * y
# y_cap = A * x = A * (A^t * A)^-1 * A^t * y = P * y
# P = A * (A^t * A)^-1 * A^t
P = A_mat @ inv((A_mat.T)@A_mat)@(A_mat.T)
# 1.5
# If A is a projection matrix, then A^2 = A
P_square = P @ P

y_cap_orth_vals = P @ y_vals
print(np.isclose(y_cap_orth_vals, y_cap_vals, atol=1e-4))


# 1.6
plt.plot(x_vals, y_vals, label="given", linestyle="dashed")
plt.plot(x_vals, y_cap_vals, label="calc")
plt.plot(x_vals, y_cap_orth_vals, label="ortho-calc", linestyle="dashed")
plt.xlabel("x-values")
plt.ylabel("f(x)")
plt.title("Given values vs Linear regression values")
plt.legend()
plt.grid()

plt.show()
print(f"We observe that both ways of calculating the y_cap results in the same graph")
print(f"Hence both methods can be used to perform linear regression")

# 1.6
plt.plot(x_vals, y_vals, label="given", linestyle="dashed")
plt.plot(x_vals, y_cap_vals, label="calc")
plt.plot(x_vals, y_cap_orth_vals, label="ortho-calc", linestyle="dashed")
plt.xlabel("x-values")
plt.ylabel("f(x)")
plt.title("Given values vs Linear regression values")
plt.legend()
plt.grid()

plt.show()
print(f"We observe that both ways of calculating the y_cap results in the same graph")
print(f"Hence both methods can be used to perform linear regression")

# 1.8
# reflection matrix R = 2P - I
R = 2 * P - np.eye(len(P))
# verify reflection R^2 = I
print(np.allclose(R @ R, np.eye(len(P))))

# --------------------- Task 2 ---------------------------
A_task2 = np.array([[1, 3, 2],[-3, 4, 3],[2, 3, 1]])
# calculated on paper, can also use eigenvalues, eigenvectors = np.linalg.eig(A_task2)
lambda_task2 = np.array([-1, 3, 4])

T_task2 = np.array([[-1/19, 1, 1],[-12/19, 0, 1/3],[1, 1, 1]])

# using lambda values to create a D matrix
D_test = np.array([[-1, 0, 0],[0, 3, 0],[0, 0, 4]])

# eigenvectors of given A matrix, calc on paper, can use np.linalg.eig(A_task2) also
e1_task2 = np.array([-1/19, -12/19, 1])
e2_task2 = np.array([1, 0, 1])
e3_task2 = np.array([1, 1/3, 1])

# 2.1
def get_z_0_in_c_vals(z_0):
    z_0_in_c = np.linalg.solve(T_task2, z_0)
    return z_0_in_c

def calc_zk(z_0, k):
    #print(f"1st : {z_0[0] * (lambda_task2[0]**k) * e1_task2}")
    #print(f"2nd : {z_0[1] * (lambda_task2[1]**k) * e2_task2}")
    #print(f"3rd : {z_0[2] * (lambda_task2[2]**k) * e3_task2}")
    z_k = z_0[0] * (lambda_task2[0]**k) * e1_task2 + \
          z_0[1] * (lambda_task2[1]**k) * e2_task2 +  \
          z_0[2] * (lambda_task2[2]**k) * e3_task2
    return z_k
z_0_given = np.zeros((3,3), dtype=float)
z_0_given[:][0] = np.array([8,3,12])
z_0_given[:][1] = np.array([1/19,-12/19,1])
z_0_given[:][2] = np.array([1/19,12/19,-1])

z_k_limit = np.zeros((3,3), dtype=float)

for given_inp_idx in range(3):
    z_0_in_c = get_z_0_in_c_vals(z_0_given[:][given_inp_idx])    
    z_k_last = z_0_in_c
    i = 1
    while (True):
        z_k_current = calc_zk(z_0_in_c, i)
        if (np.allclose(z_k_current, z_k_last) == True):
            z_k_limit[:][given_inp_idx] = z_k_current
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | The limit is : {z_k_current}")
            break
        z_k_last = z_k_current
        i = i + 1
        if (i > 100):
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | There is no limit")
            break

# To see the sequence behaviour run the below code
# change the column index in z_0_given to see different sequences

#z_0_in_c = get_z_0_in_c_vals(z_0_given[:][0])
#for i in range(30):
#    z_k_test1 = calc_zk(z_0_in_c, i)
#    print(f"final answer :{i} | : {z_k_test1}")
# print(f"----------------------------------------------")

# 2.2
def calc_vk(z_0, k):
    #print(f"1st : {z_0[0] * (lambda_task2[0]**k) * e1_task2}")
    #print(f"2nd : {z_0[1] * (lambda_task2[1]**k) * e2_task2}")
    #print(f"3rd : {z_0[2] * (lambda_task2[2]**k) * e3_task2}")
    z_k = calc_zk(z_0, k)
    v_k = z_k * (1/np.linalg.norm(z_k))
    return v_k

v_k_limit = np.zeros((3,3), dtype=float)

for given_inp_idx in range(3):
    z_0_in_c = get_z_0_in_c_vals(z_0_given[:][given_inp_idx])
    v_k_last = calc_vk(z_0_in_c, 0)
    i = 1
    while (True):
        v_k_current = calc_vk(z_0_in_c, i)
        if (np.allclose(v_k_current, v_k_last) == True):
            v_k_limit[:][given_inp_idx] = v_k_current
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | The limit is : {v_k_current}")
            break
        v_k_last = v_k_current
        i = i + 1
        if (i > 100):
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | There is no limit")
            break
        
        
# To see the sequence behaviour run the below code
# change the column index in z_0_given to see different sequences

z_0_in_c = get_z_0_in_c_vals(z_0_given[:][0])
for i in range(50):
    v_k_test1 = calc_vk(z_0_in_c, i)
    print(f"final answer :{i} | : {v_k_test1}")
print(f"----------------------------------------------")

# 2.3
def calc_qk(z_0,k):
    v_k = calc_vk(z_0, k)
    q_k = (v_k.T) @ A_task2 @ v_k
    return q_k

q_k_limit = np.array([0,0,0])

for given_inp_idx in range(3):
    z_0_in_c = get_z_0_in_c_vals(z_0_given[:][given_inp_idx])
    q_k_last = calc_qk(z_0_in_c,0)
    i = 1
    while (True):
        q_k_current = calc_qk(z_0_in_c, i)
        if (np.allclose(q_k_current, q_k_last) == True):
            q_k_limit[given_inp_idx] = q_k_current
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | The limit is : {q_k_current}")
            break
        q_k_last = q_k_current
        i = i + 1
        if (i > 100):
            print(f"input sequence : {z_0_given[:][given_inp_idx]} | The iteration index : {i} | There is no limit")
            break

# To see the sequence behaviour run the below code
# change the column index in z_0_given to see different sequences

#z_0_in_c = get_z_0_in_c_vals(z_0_given[:][0])
#for i in range(24):
#    q_k_test1 = calc_qk(z_0_in_c, i)
#print(f"final answer :{i} | : {q_k_test1}")


print(z_k_limit)
print(f"-------------------------")
print(v_k_limit)
print(f"-------------------------")
print(q_k_limit)

# 2.4
# TBD

# 2.5
# The default value for absolute tolerance in 1e-8 in python for np.allclose()
# So the iteration indexes obtained in task 2.2 are the number of iterates needed

# 2.6 - can do this until 9, need to figure out from 9 to 14
atol_list_v =  [10**(-n) for n in range(1,9)]
atol_list_q =  [10**(-n) for n in range(1,10)]
num_iter_v = np.zeros((2,len(atol_list_v)), dtype=float)
num_iter_q = np.zeros((2,len(atol_list_q)), dtype=float)

for given_inp_idx in range(2):
    z_0_in_c = get_z_0_in_c_vals(z_0_given[:][given_inp_idx])
    v_k_last = calc_vk(z_0_in_c, 0)
    for atol_idx in range(len(atol_list_v)):
        i = 1
        while (True):
            v_k_current = calc_vk(z_0_in_c, i)
            if (np.allclose(v_k_current, v_k_last, atol=atol_list_v[atol_idx]) == True):
                num_iter_v[given_inp_idx][atol_idx] = i
                break
            v_k_last = v_k_current
            i = i + 1
            if (i > 200):
                print(f"reached {i} iterations for {atol_list_v[atol_idx]} | v calc")
                break

for given_inp_idx in range(2):
    z_0_in_c = get_z_0_in_c_vals(z_0_given[:][given_inp_idx])
    q_k_last = calc_qk(z_0_in_c,0)
    for atol_idx in range(len(atol_list_v)):
        i = 1
        while (True):
            q_k_current = calc_qk(z_0_in_c, i)
            if (np.allclose(q_k_current, q_k_last, atol=atol_list_q[atol_idx]) == True):
                num_iter_q[given_inp_idx][atol_idx] = i
                break
            q_k_last = q_k_current
            i = i + 1
            if (i > 100):
                print(f"reached {i} iterations for {atol_list_q[atol_idx]}")
                break

print(atol_list_v)
print(num_iter_v)
print(num_iter_q)

plt.figure(figsize=(7,5))
plt.semilogx(atol_list_v, num_iter_v[0][:], label=r'$y = \log(x)$')
plt.semilogx(atol_list_v, num_iter_v[1][:], label=r'$y = \log(x)$')

plt.xlabel("tolerance values (log scale)")
plt.ylabel("num of iterations")
plt.title("Plot of num of iter vs tolerance")
plt.legend()
plt.grid(True)

plt.show()

print("float32:", np.float32(1.123456789123456789))
print("float64:", np.float64(1.123456789123456789))
print("float128:", np.float128(1.123456789123456789))


