import numpy as np
import pandas as pd

from numba import njit, prange

@njit(parallel=True)
def shares_compute(K, D, S, sigma_a, sigma_e):
    '''A function that computes the expenditure shares from given imputs.
    
    Note that `E` denotes the number of exporters, `I` the number of importers
    and `T` the number of time periods.
    
    Parameters
    ----------
    K: ndarray
        An E*1 vector of ln(GDP_i/GDP_1) values (where 1=USA) of relative
        GDP values in the pre-sample period.
    D: ndarray
        An E*I*T vector of delta_{ji,t} values of effective factor prices
    S: ndarray
        An M*2 vector of (alpha_S, epsilon_S) sample values
    sigma_a: float
    sigma_e: float
        
    Returns
    -------
    X: ndarray
        An E*I*T vector of x_{ji,t} values of expenditure shares
    '''
    
    E, I, T = D.shape 
    M = S.shape[0]
    
    X = np.zeros((E, I, T))
    
    for j in prange(E):
        for i in prange(I):
            for t in prange(T):
                
                #Loop over all simulated values here
                for s in prange(M):
                    up = np.exp(sigma_a*S[s,0] * K[j] + D[j,i,t] * S[s,1]**sigma_e)
                    down = 1.0
                    
                    for l in prange(E-1):
                        down += np.exp(sigma_a*S[s,0] * K[l] + D[l,i,t] * S[s,1]**sigma_e)
                    
                    X[j,i,t] += (1/M) * up / down
    
    return X

def shares_converge(X, K, D, S, sigma_a, sigma_e, tol=1e-3):
    
    error = 1
    
    while error > tol:
        Xs = shares_compute(K, D, S, sigma_a, sigma_e)
        error = np.abs(X-Xs).sum()
        
        D = D + 3*np.log(X/np.maximum(Xs,1e-7))
        
        # D for as exporter always 0
        D[-1, :, :] = 0
        
        #print(f'error:{error:.2f}, max_d:{np.abs(X-Xs).sum():.2f}, max_D:{np.abs(D).max():.2f}, max_Xs:{np.abs(Xs).max():.2f}')
        
    return Xs, D