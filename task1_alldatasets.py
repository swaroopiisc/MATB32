# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 19:16:03 2025

@author: swaro
"""

import pandas as pd
import numpy as np
from scipy.optimize import fmin
from numpy.linalg import inv
#from numpy.linalg import pinv
import matplotlib.pyplot as plt

data_as_pandas_df_dat1 = pd.read_csv("regression_1.dat", delimiter=",")
data_as_pandas_df_dat2 = pd.read_csv("regression_2.dat", delimiter=",")
data_as_pandas_df_dat2 = data_as_pandas_df_dat2.sort_values(by="x", ascending=True)

def ycap_minus_y(args):
    a, b = args
    y_cap = a * x_vals + b
    error = np.sum((y_cap - y_vals)**2)
    return error

def task1_entire(x_vals, y_vals):
    # assuming a_start = 1, then b_start = y-x This needs 28 iterations
    a_start = 1
    b_start = y_vals[0] - x_vals[0]
    intial_guess = [a_start, b_start]
    
    optimal_guess = fmin(ycap_minus_y, intial_guess)
    
    n = 2 #2 unknowns
    m = len(x_vals) #can also use y_vals
    A_mat = np.zeros([m,n], dtype = 'float')
    A_mat[:,0] = x_vals
    A_mat[:,1] = [1 for p in range(m)]
    calc_guess = inv((A_mat.T)@A_mat)@(A_mat.T)@y_vals
    # 1.1 a)
    print(f"Using scipy.optimize.fmin")
    print(f"a = {optimal_guess[0]} | b = {optimal_guess[1]}")
    print(f"Using normal eqns and numpy")
    print(f"a = {calc_guess[0]} | b = {calc_guess[1]}")
    # 1.2
    y_cap_vals = optimal_guess[0] * x_vals + optimal_guess[1]
    figure1 = plt.figure(figsize=(8, 8))
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
    
    x_chosen = 1
    y_opt_a_b = optimal_guess[0] * x_chosen + optimal_guess[1]
    def err_fn(a,b):
        y_rand_a_b = a * x_chosen + b
        return abs(y_rand_a_b - y_opt_a_b)
    
    err_for_opt = err_fn(optimal_guess[0], optimal_guess[1])
    
    X, Y = np.meshgrid(a_vals, b_vals)
    Z = err_fn(X,Y)
    figure2 = plt.figure(figsize=(8, 8))
    ax = figure2.add_subplot(111, projection='3d')
    error_surface = ax.plot_surface(X, Y, Z, cmap='magma')
    cbar = figure2.colorbar(error_surface, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label("Intensity (Z-values)")
    ax.scatter(optimal_guess[0], optimal_guess[1], err_for_opt, color='green', s=200, marker='s', label="Marked Point")
    ax.view_init(elev=15, azim = 75)
    ax.set_xlabel("a values")
    ax.set_ylabel("b values")
    ax.set_zlabel("error = ||y_cap - y||")
    ax.set_title("surface plot of error")
    
    P = A_mat @ inv((A_mat.T)@A_mat)@(A_mat.T)
    P_square = P @ P
    
    y_cap_orth_vals = P @ y_vals
    print(np.isclose(y_cap_orth_vals, y_cap_vals, atol=1e-4))
    
    plt.show()
    print(f"We observe that both ways of calculating the y_cap results in the same graph")
    print(f"Hence both methods can be used to perform linear regression")
    
    # 1.8
    # reflection matrix R = 2P - I
    R = 2 * P - np.eye(len(P))
    # verify reflection R^2 = I
    print(np.allclose(R @ R, np.eye(len(P))))
    refl_vec = R @ y_vals
    
    # 1.6
    figure3 = plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, label="given", linestyle="dashed")
    plt.plot(x_vals, y_cap_vals, label="calc")
    plt.plot(x_vals, y_cap_orth_vals, label="ortho-calc", linestyle="dashed")
    plt.plot(x_vals, refl_vec, label="reflected vec")
    plt.xlabel("x-values")
    plt.ylabel("f(x)")
    plt.title("Given values vs Linear regression values")
    plt.legend()
    plt.grid()

# data 1 - set 1
x_vals = data_as_pandas_df_dat1["x"]
y_vals = data_as_pandas_df_dat1["y"]
task1_entire(x_vals, y_vals)
# data 2 - set 1
x_vals = data_as_pandas_df_dat2["x"]
y_vals = data_as_pandas_df_dat2["y1"]
task1_entire(x_vals, y_vals)
# data 2 - set 2
x_vals = data_as_pandas_df_dat2["x"]
y_vals = data_as_pandas_df_dat2["y2"]
task1_entire(x_vals, y_vals)
# data 2 - set 3
x_vals = data_as_pandas_df_dat2["x"]
y_vals = data_as_pandas_df_dat2["y3"]
task1_entire(x_vals, y_vals)
