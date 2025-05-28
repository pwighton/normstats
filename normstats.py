import random
import numpy as np
from numpy.linalg import inv
from scipy.stats import t, norm
from nct import nct_cdf_solve_for_nc
import logging

logger = logging.getLogger(__name__)

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
    logger.debug(f'X.shape: {X.shape}')

    n = X.shape[1]
    logger.debug(f'n: {n}')
    
    if X_mean is None:
        X_mean = np.mean(X, axis=1)
        # reshape from (k,) to (k, 1) so broadcasting will work
        X_mean = X_mean[:, np.newaxis]
    if X_std is None:
        X_std = np.std(X, axis=1)
        # reshape from (k,) to (k, 1) so broadcasting will work
        X_std = X_std[:, np.newaxis]

    logger.debug(f'X_mean.shape: {X_mean.shape}')
    logger.debug(f'X_std.shape: {X_std.shape}')

    Z = (X - X_mean)/X_std
    Z_1 = np.vstack((Z, np.ones((1,n))))

    logger.debug(f'Z.shape: {Z.shape}')
    logger.debug(f'Z_1.shape: {Z_1.shape}')
    
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
    X_are_ints: Boolean
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
    logger.debug(f'n: {n}')
    logger.debug(f'X_are_ints: {X_are_ints}')

    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1
    
    logger.debug(f'm: {m}')
    logger.debug(f'k: {k}')
    
    if X_are_ints:
        X = np.random.randint(X_range[0], X_range[1], size=(k, n))
    else:
        X = np.random.uniform(X_range[0], X_range[1], size=(k, n))
    logger.debug(f'X.shape: {X.shape}')

    if S_YdotX is None:
        S_YdotX = np.tile(1.0, (m))
    assert(S_YdotX.shape == (m,)) 
    logger.debug(f'S_YdotX.shape: {S_YdotX.shape}')
    
    Z_1, _, _, _ = z_normalize(X)

    scale_array = np.tile(S_YdotX, (n,1)).T
    epsilon = np.random.normal(loc=0, scale=scale_array, size=(m,n))
    logger.debug(f'epislon.shape: {epsilon.shape}')
    
    Y = B @ Z_1 + epsilon
    logger.debug(f'Y.shape: {Y.shape}')
    
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
    S_YdotX_estimate: (m x 1) array
        The estimated standard deviation of the residual
    R : (k x k) array
        The inverse of the correlation matrix (R = (ZZ^T)^-1)
    X_mean: (k x 1) array
        The means of the independent variables
    X_std: (k x 1) array
        The standard deviations of the independent variables.
    '''
    logger.debug(f'X.shape: {X.shape}')
    logger.debug(f'Y.shape: {Y.shape}')
    
    Z_1, Z, X_mean, X_std = z_normalize(X)
    R = inv(Z @ Z.T)
    B_estimate = Y @ Z_1.T @ inv(Z_1 @ Z_1.T)
    
    Y_estimate = B_estimate @ Z_1
    S_YdotX_estimate = np.std(Y - Y_estimate, axis=1)
    # reshape from (m,) to (m, 1) so broadcasting will work
    S_YdotX_estimate = S_YdotX_estimate[:, np.newaxis]
    
    logger.debug(f'B_estimate.shape: {B_estimate.shape}')
    logger.debug(f'S_YdotX_estimate.shape: {S_YdotX_estimate.shape}')
    logger.debug(f'R.shape: {R.shape}')
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
        resdual is used for all dependent variables and all subjects
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
    logger.debug(f'X_are_ints: {X_are_ints}')
    logger.debug(f'n: {n}')
        
    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1

    logger.debug(f'm: {m}')
    logger.debug(f'k: {k}')
    
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

def single_subject_eval(x_obs, y_obs, B, R, S_YdotX, n, X_mean, X_std, ci_alpha=0.05):
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
    ci_alpha: float
        The significance level when computing the confidence interval of the percentile estimates
        
    Produce:
    -------------------
    p : A (m x 1) array
        The percentile estimates of the dependent variables, relative to the normative set
    p_ci: A (m x 2) array
        The lower and upper `ci_alpha`-confidence intervals for p
    '''
    
    # The number of dep vars
    m = B.shape[0]
    # The number of ind vars
    k = B.shape[1] - 1

    logger.debug(f'm: {m}')
    logger.debug(f'k: {k}')
    
    # z-normalize the subjects
    z_obs1, z_obs, _, _ = z_normalize(x_obs, X_mean, X_std)
    y_estimate = B @ z_obs1
    
    r_A = np.sum(np.diag(R) * (z_obs.T ** 2))
    logger.debug(f'r_A: {r_A}')
    
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
    logger.debug(f'r_B: {r_B}')
    
    # Compute S_{N+1}
    S_Nplus1 = S_YdotX * np.sqrt(1 + 1/n + 1/(n-1)*r_A + 2/(n-1)*r_B)
    logger.debug(f'S_Nplus1: {S_Nplus1}')
    
    # Compute the t-statistic
    t_diff = (y_obs - y_estimate)/S_Nplus1
    logger.debug(f't_diff: {t_diff}')

    # Compute the percentile estimate
    # This is "a point estimate of the percentage of the control population that
    # would exhibit a larger discrepancy"
    p = t.cdf(x=t_diff, df=n-k-1)
    
    # Compute confidence intervals for p
    #tau = np.where(y_estimate > y_obs, 1, -1)
    #logger.debug(f'tau: {tau}')
    
    c = (y_obs - y_estimate) / S_YdotX
    logger.debug(f'c: {c}')
    
    theta = 1/n + (r_A + 2 * r_B)/(n-1)
    logger.debug(f'theta: {theta}')
    
    #nct_stat = tau * c / np.sqrt(theta)
    nct_stat = c / np.sqrt(theta)
    
    delta_L = np.zeros(p.shape)
    delta_U = np.zeros(p.shape)
    p_ci_lower = np.zeros(p.shape)
    p_ci_upper = np.zeros(p.shape)

    logger.debug(f'df: {n-k-1}')
    for i in range(m):
        logger.debug(f'nct_stat:  {nct_stat[i][0]}')
        delta_L[i][0] = nct_cdf_solve_for_nc(nct_stat[i][0], n-k-1, 1-ci_alpha/2)
        delta_U[i][0] = nct_cdf_solve_for_nc(nct_stat[i][0], n-k-1, ci_alpha/2)
        p_ci_lower[i][0] = norm.cdf(delta_L[i][0]*np.sqrt(theta))
        p_ci_upper[i][0] = norm.cdf(delta_U[i][0]*np.sqrt(theta))

    p_ci = np.hstack((p_ci_lower, p_ci_upper))
    return p, p_ci
