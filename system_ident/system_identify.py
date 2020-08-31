__author__ = "Mihir Juneja"
__version__ = "1.0"
__email__ = "mihirjuneja17@gmail.com"

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

## value used in equation 4
MU = 0.001
NMAX = 800

# ------------------------ FUNCTIONS ---------------------------------------

# Function: read_data()
#
# Input: File name
# Returns: Input u, Desired output y0
#
# Description: This function reads the data file. It then assigns each column a name and
# finally returns the columns for the input 'u' as well as the desired output 'y0'
#
def read_data(filename):
    f = pd.read_csv(filename, header=None)
    # assigning column names
    f.columns = ['i', 'len_u', 'u', 'y0']
    return f['u'], f['y0']

# Function: sum_of_prod()
#
# Inputs: Coefficients, Input u, Order of filter, time index
# Returns: computed sum of products (eqn 2)
#
# Description: This function is called by FIR function (equation 2).
# It iterates over order and computes sum of products of input u and coefficents h
#
def sum_of_prod(h, u, M, n):
    # Initialize sop
    sop = 0.0
    for k in range(M):
        # loop over k till k <= n
        if (k <= n):
            sop += h[k, n] * u[n - k]
    return sop

# Function: v_sum_of_prod (v_ stands for verify)
#
# Inputs: Converged Coefficients, Input u, Order of filter, time index
# Returns: computed sum of products (eqn 2)
#
# Description: This function is called by v_FIR function (equation 2).
# It iterates over order and computes sum of products of input u and coefficents h
# In this case converged "h" are a 1D array and not 2D like sum_of_prod()
# "h" here does NOT depend on "n"
#
def v_sum_of_prod(h, u, M, n):
    # Initialize sop
    sop = 0.0
    for k in range(M):
        # loop over k till k <= n
        if (k <= n):
            sop += h[k] * u[n - k]
    return sop

# Function: FIR()
#
# Inputs: Input u, Desired output y0, Order of filter M
# Returns: Computed output y1, error signal e
#
# Description: This function implements equations for h1[k,n+1] and y1[n] and e[n].
# It computes the value of the equations 2,3 and 4. Finally it returns y1, h1 and error.
# It also allows choosing initial h1 as RANDOM or ZEROS
#
def FIR(u_in, y0_in, M, init_coef):
    #
    MAX = len(u_in)
    # Input u
    u = np.empty(NMAX)
    # Desired output y0
    y0 = np.empty(NMAX)

    u[0:NMAX] = u_in
    y0[0:NMAX] = y0_in
    # Choosing initial coeffecients
    if init_coef == 'RANDOM':
        h1 = np.random.normal(0, 2, size=(M, MAX))
    else:
        h1 = np.zeros(shape=(M, MAX), dtype=float)

    # Initialising y1
    y1 = np.zeros(MAX, dtype=float)
    #Initialising error
    e = np.zeros(MAX, dtype=float)
    for n in range(MAX - 1):
        ## equation 2 calculating y1
        y1[n] = sum_of_prod(h1, u, M, n)

        ## equation 3 calculating the error
        e[n] = y0[n] - y1[n]

        for k in range(M):
            ## equation 4 calculating coefficeints h1
            if (k <= n):
                h1[k, n + 1] = h1[k, n] + MU * e[n] * u[n - k]

    # Above loop was run only till MAX-1 because equation for h1 has n+1
    n = MAX - 1
    y1[n] = sum_of_prod(h1, u, M, n)
    e[n] = y0[n] - y1[n]

    return y1, e, h1

# Function: v_FIR() (v_ stands for verify)
#
# Inputs: Input u, Desired output y0, Order of filter M, converged h1
# Returns: Computed output y1, error signal e
#
# Description: This function implements equations for y1[n] and e[n].
# It computes the value of the equations 2,3 and 4. Finally it returns y1 and error.
# The main difference wrt FIR() is that h1 is fixed and not updated for each n
#
def v_FIR(u_in, y0_in, M, h1):
    MAX = len(u_in)
    u = np.empty(NMAX)
    y0 = np.empty(NMAX)

    u[0:NMAX] = u_in
    y0[0:NMAX] = y0_in

    # initialising y1 and e
    y1 = np.zeros(MAX, dtype=float)
    e = np.zeros(MAX, dtype=float)
    for n in range(MAX):
        y1[n] = v_sum_of_prod(h1, u, M, n)
        e[n] = y0[n] - y1[n]

    return y1, e

# Function: compute_ens_avg()
#
# Inputs: M, init_coef
# Returns: E_h1_mean_conv
#
# Description: This function implements equations for calculating the ensemble averages
# It runs for 50 iterations and calls the function FIR().
# It calculates ensemble average for error, h1 and finally converged h1 values.
# It also plots the converged h1 which is ensemble averaged first and then time averaged over the last 100 samples.
#
def compute_ens_avg(M, init_coef):
    # no. of iterations
    ITER = 50
    NMAX = 800

    # initialising ensemble averages for h1 and error
    E_err = np.zeros(shape=(ITER, NMAX), dtype=float)
    E_h1  = np.zeros(shape=(ITER, M, NMAX), dtype=float)

    for i in range(ITER):
        print("Iteration ", i + 1, 'in progress')
        L = i * NMAX
        H = L + NMAX
        y1, e, h1 = FIR(u[L:H], y0[L:H], M, init_coef)
        E_err[i, :] = np.square(e)

        for k in range(M):
            for n in range(NMAX):
                E_h1[i, k, n] = h1[k, n]

    E_err_mean = np.mean(E_err, axis=0)

    # error avg plot
    plt.plot(E_err_mean, 'g-')
    plt.title('Ensemble Avg (Error)    Init coeff =  ' + init_coef + ' , Order = ' + str(M))
    f_name = 'ens_avg_err_m' + str(M) + '_' + init_coef + '.pdf'
    plt.savefig(f_name)
    plt.show()

    E_h1_mean = np.mean(E_h1, axis=0)
    E_h1_mean_conv = np.mean(E_h1_mean[:,700:800], axis=1)
    #E_h1_mean_conv = np.mean(E_h1_mean, axis=1)
    print('Converged h1, averaged over 100 last time samples = ', E_h1_mean_conv)

    print('Variance of ens avg h1 = ', np.var(E_h1[:, :, 799]))

    # Plotting the converged h1 values.
    plt.figure(figsize=(8, 5))
    plt.title('Converged H1 (Ens avg + Time avg over last 100 samples)' + '       Order = ' + str(M))
    plt.plot(E_h1_mean_conv, 'b-')
    plt.savefig('conv_h1_100_samples.pdf')
    plt.show()

    return E_h1_mean_conv

# Function: verify_filter()
#
# Inputs: M, h1_converged
# Returns: Nil
#
# Description: Calls the v_FIR function and runs it for fixed converged h1 values.
# It allows choosing the data set to be run & plotted, which can be adjusted by changing the values of L and H
# This function plots y1, y0 and error for converged h1.
#

def verify_filter(M, h1_converged):
    NMAX = 800

    L = 0
    H = 800
    # iteration no.
    iter = 1

    y1, e = v_FIR(u[L:H], y0[L:H], M, h1_converged)

    #initialising y0_temp
    y0_temp = np.empty(NMAX)
    y0_temp[0:800] = y0[L:H]

    plt.figure(figsize=(10,8))
    ## subplot with y1,y0 outputs and error
    plt.subplot(2, 1, 1)
    plt.plot(y1, 'b-')
    plt.plot(y0_temp, 'g-')
    plt.gca().legend(('y1', 'y0'))
    plt.title('Outputs y1 and y0 for converged H1  (Iteration = ' + str(iter) + ')                    Order = ' + str(M))

    ## Plotting error
    plt.subplot(2, 1, 2)
    plt.plot(e, 'r.-')
    plt.title('error')
    plt.savefig('converged_plot_Iter_' + str(iter) + '.pdf')
    plt.show()

# ------------------------------------------ MAIN ------------------------

np.set_printoptions(precision=2)

## Read input data
u, y0 = read_data('data_new.csv')

## order of the filter
M = 10

######## You may want to enter order interactively, uncomment the following
#order_str = input("Enter Filter Order: ")
#M = int(order_str)

## Initialize coefficient for first time sample
init_coeff = 'ZEROS'

## calculating the ensemble averages
h1_converged = compute_ens_avg(M, init_coeff)

## Plotting y1,y0 and error for converged h1
verify_filter(M, h1_converged)