import logging
import scipy as sp

logger = logging.getLogger(__name__)

def nct_cdf_solve_for_nc(x, df, cdf, bounds=None, num_tries=100):
    """
    Solve for the non-centrality parameter (nc) in a cumulative non-central T
    distribution given the point (x), degrees of freedom (df), and the CDF value
    (cdf).
    
    Parameters:
    -----------
    x : float
        The point at which the CDF is evaluated
    df : int
        Degrees of freedom
    cdf : float
        The target value of the cumulative distribution function (between 0 and 1)
    bounds : tuple, optional
        Lower and upper bounds for the solution
        
    Returns:
    --------
    float
        The non-centrality parameter nc such that T_{df,nc}(x) = cdf
    """
    
    def objective(nc):
        return sp.stats.nct.cdf(x, df, nc) - cdf

    if bounds == None:
        bounds = (-10,10)

    result = None
    logger.debug(f'x: {x}')
    logger.debug(f'df: {df}')
    logger.debug(f'cdf: {cdf}')
    
    for i in range(num_tries):
        try:
            logger.debug(f'bounds: {bounds}')
            result = sp.optimize.brentq(objective, bounds[0], bounds[1], rtol=1e-10)
            break
        except ValueError as e:
            error_msg = str(e)
            if "f(a) and f(b) must have different signs" in error_msg:
                # This means the bounds were too tight
                logger.debug('different signs exception caught; loosening bounds')
                bounds = tuple(x*2.0 for x in bounds)
                continue
            elif "is NaN; solver cannot continue" in error_msg:
                # This means the bounds are too loose
                logger.debug('function value at x=%f is NaN exception caught; tightening bounds')
                # We divide by a differnt number than what we multiply by to prevent an infite loop
                bounds = tuple(x/1.75 for x in bounds)
                continue
            else:
                raise e
    if result == None:
      logger.error('Could not find a solution')
      raise ValueError('Could not find a solution')
      
    logger.debug(f'result: {result}')
    logger.debug(f'objective: {objective(result)}')
    return result
