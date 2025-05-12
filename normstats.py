import random
import numpy as np
from numpy.linalg import inv
from scipy.stats import t

def z_normalize(X, X_mean=None, X_std=None):
    """
    Z-normalize the independent variables

    Given:
    -------------------
    X : (k x n) array
        The independent variables to z-normalize
    X_mean : (k x 1) array | None
        The means of the independent variables.  If None, they are estimated
    X_std : (k x 1) array | None
        The standard deviations of the independent variables.  If None, they are estimated
        
    Produce:
    -------------------
    Z_1 : (k+1 x n) array
        The z-normalized independent variables, with a 1's vector appended
    Z : (k x n) array
        The z-normalized independent variables, without a 1's vector appended
    X_mean: (k x 1) array
        The means of the independent variables
    X_std: (k x 1) array
        The standard deviations of the independent variables.
    """
    n = X.shape[1]
    
    if X_mean is None:
        X_mean = np.mean(X)
    if X_std is None:
        X_std = np.std(X)
    
    Z = (X - X_mean)/X_std
    Z_1 = np.vstack((Z, np.ones((1,n))))
    
    return Z_1, Z, X_mean, X_std

# Generate synthetic data
#  - B: The actual model parameters; should be a 1xk array
#  - n: The number of synthetic subjects to generate
#  - k: The number of independent variables
#  - S_YdotX: The standard deviation of the residual
#  - X_range: The range of values for the ind vars, should be a 2-length tuple
#  - X_are_ints: When generating ind vars for the subject, should they be ints or floats?

def gen_synth_norm_data(B,n=100,S_YdotX=None,X_range=(0,10),X_are_ints=True):
    '''
    Generate synthetic normative data

    Given:
    -------------------
    B : (m x k+1) array
        The actual model parameters
    n : int
        The number of synthetic subjects to generate
    S_YdotX : m-length 1d array
        The standard deviation of the residual
    X_range: 2-length tuple
        The range of values for the ind vars
    X_are_int: Boolean
        If True, generated independent variables are Ints
        If False, generated independent variables are Floats

    Produce:
    -------------------
    X : A (k x n) array
        The sythetically generated array of independent variables
    Y : A (m x n) array
        The dependent variables
    epsilon : A (m x n) array
        The residuals
    '''

    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1
    
    if X_are_ints:
        X = np.random.randint(X_range[0], X_range[1], size=(k, n))
    else:
        X = np.random.uniform(X_range[0], X_range[1], size=(k, n))

    if S_YdotX is None:
        S_YdotX = np.tile(1.0, (m))
    assert(S_YdotX.shape == (m,)) 
    
    Z_1, _, _, _ = z_normalize(X)

    scale_array = np.tile(S_YdotX, (n,1)).T
    epsilon = np.random.normal(loc=0, scale=scale_array, size=(m,n))
    Y = B @ Z_1 + epsilon

    return X, Y, epsilon

def estimate_model_params(X,Y):
    '''
    Estimate model parameters from a normative set of independent and dependent variables

    Given:
    -------------------
    X : A (k x n) array
        The independent variables
    Y : A (m x n) array
        The dependent variables
        
    Produce:
    -------------------
    B_estimate: (m x k+1) array
        The estimated model parameters
    S_YdotX_estimate: (m,) array
        The estimated standard deviation of the residual
    R : (k x k) array
        The inverse of the correlation matrix (R = (ZZ^T)^-1)
    X_mean: (k x 1) array
        The means of the independent variables
    X_std: (k x 1) array
        The standard deviations of the independent variables.
    '''
    Z_1, Z, X_mean, X_std = z_normalize(X)
    R = inv(Z @ Z.T)
    B_estimate = Y @ Z_1.T @ inv(Z_1 @ Z_1.T)
    
    Y_estimate = B_estimate @ Z_1
    S_YdotX_estimate = np.std(Y - Y_estimate, axis=1)

    return B_estimate, S_YdotX_estimate, R, X_mean, X_std

def gen_synth_sub_data(B, n, epsilon, X_range, X_mean, X_std, X_are_ints=True):
    '''
    Generate synthetic subject data, to be evaluated using model pramaters from normative data

    Given:
    -------------------
    B: (m x k+1) array
        The model parameters used to generate the independent vars
    n : int
        The number of synthetic subjects to generate
    epsilon: float | (k x n) array
        The residuals to use when generating data.  If epsilon is a single float, the same
        resdual is used for all dependend variables and all subjects
    X_range: 2-length tuple
        The range of values for the ind vars
    X_mean: (k x 1) array
        The means of the independent variables of the normative set
    X_std: (k x 1) array
        The standard deviations of the independent variables of the normative set    
    X_are_int: Boolean
        If True, generated independent variables are Ints
        If False, generated independent variables are Floats
        
    Produce:
    -------------------
    X : A (k x n) array
        The sythetically generated array of independent variables
    Y : A (m x n) array
        The dependent variables
    '''
    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1

    if isinstance(epsilon, float):
        epsilon = np.tile(epsilon, (m,n))
    assert(epsilon.shape == (m,n))

    if X_are_ints:
        X = np.random.randint(X_range[0], X_range[1], size=(k, n))
    else:
        X = np.random.uniform(X_range[0], X_range[1], size=(k, n))

    Z_1, _, _, _= z_normalize(X, X_mean, X_std)
    Y = B @ Z_1 + epsilon
    
    return X, Y

def single_subject_eval(x_obs, y_obs, B, R, S_YdotX, n, X_mean, X_std):
    '''
    Evaluate a single subject and compute percentile estimates
    
    Given:
    -------------------
    x_obs : A (k x 1) array
        The subjects's observed independent variables
    y_obs: A (m x 1) array
        The subject's observed independent variables
    B: (m x k+1) array
        The model parameters as estimated from the normative dataset
    R : (k x k) array
        The inverse of the correlation matrix as estimated from the normative dataset
    S_YdotX : (m x 1) array
        The estimated standard deviation of the residuals as estimated from the normative dataset
    n : int
        The number of subjects in the normative dataset
    X_mean: (k x 1) array
        The means of the independent variables of the normative set
    X_std: (k x 1) array
        The standard deviations of the independent variables of the normative set  
    
    Produce:
    -------------------
    p : A (m x 1) array
        The percentile estimates of the dependent variables, relative to the normative set
    t_diif: A (m x 1) array
        The t-statistics of the differences between the observed dependent variables and the predicted
        dependent variables
    '''
    
    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1
    
    # z-normalize the subjects
    z_obs1, z_obs, _, _ = z_normalize(x_obs, X_mean, X_std)
    y_estimate = B @ z_obs1
    
    r_A = np.sum(np.diag(R) * (z_obs.T ** 2))
    
    # Computing r_B:
    # ------------------
    # the upper matrix elements, excluding the diagonal
    #  - https://stackoverflow.com/a/47314816
    # - we pass `k=1` to triu_indices() to exclude diagonal elements
    uR_idx = np.triu_indices(R.shape[0], k=1)

    # This is the r_{i,j} term of the equation for r_B
    r_i_j = R[uR_idx[0],uR_idx[1]]
    # This is the z_{obs,i} term of the equation for r_B
    z_obs_i = z_obs[uR_idx[0]].T
    # This is the z_{obs,j} term for the equation for r_B
    z_obs_j = z_obs[uR_idx[1]].T

    r_B = np.sum(r_i_j * z_obs_i * z_obs_j)

    # Compute S_{N+1}
    S_Nplus1 = S_YdotX * np.sqrt(1 + 1/n + 1/(n-1)*r_A + 2/(n-1)*r_B)
    
    # Compute the t-statistic
    t_diff = (y_obs - y_estimate).T/S_Nplus1
    
    # Compute the percentile estimate, since this is a perfectly median subject,
    # It should be ~0.5.  The larger n, the closer this should be to 0.5
    p = t.cdf(x=t_diff, df=n-k)

    return p, t_diff
