import math
import numpy as np
from typing import Tuple, Optional

def calc_min_variance_portfolio(
    return_vector: np.matrix, 
    stddev_vector: np.matrix, 
    correlation_matrix: np.matrix, 
    target_return: float
) -> Tuple[np.matrix, float]:
    """
    Calculates the minimum variance portfolio that achieves the target return, if possible.
    """
    MU = return_vector
    R = correlation_matrix
    S = np.matrix(np.diagflat(stddev_vector))
    COV = S * R * S
    ONE = np.matrix((1,) * COV.shape[0]).T
    A, B, C = ONE.T * COV.I * ONE, MU.T * COV.I * ONE, MU.T * COV.I * MU
    a, b, c = float(A), float(B), float(C)
    LAMBDA, GAMMA = (a * target_return - b) / (a * c - (b * b)), ((c - b * target_return) / ((a * c) - (b * b)))
    WSTAR = COV.I * ((LAMBDA * MU) + (GAMMA * ONE))
    STDDEV = math.sqrt(WSTAR.T * COV * WSTAR)
    return WSTAR, STDDEV

def calc_max_return_portfolio(
    return_vector: np.matrix, 
    stddev_vector: np.matrix, 
    correlation_matrix: np.matrix, 
    target_variance: float
) -> Tuple[Optional[np.matrix], Optional[float], Optional[float]]:
    """
    Calculates the maximum return portfolio that achieves the target variance, if possible.
    """
    last_return, last_allocation, last_stddev = None, None, None
    for target_return in np.linspace(float(min(return_vector)), float(max(return_vector)), 500):
        this_allocation, this_stddev = calc_min_variance_portfolio(return_vector, stddev_vector, correlation_matrix, target_return)
        if this_stddev > target_variance:
            return (last_allocation, last_stddev, last_return)
        last_allocation, last_stddev, last_return = this_allocation, this_stddev, target_return
    return None, None, None

if __name__ == "__main__":
    return_vector = np.matrix((0.05, 0.07, 0.15, 0.27)).T
    stddev_vector = np.matrix((0.07, 0.12, 0.30, 0.60)).T
    correlation_matrix = np.matrix(((1.0, 0.8, 0.5, 0.4),
                                       (0.8, 1.0, 0.7, 0.5),
                                       (0.5, 0.7, 1.0, 0.8),
                                       (0.4, 0.5, 0.8, 1.0)))

    target_return = .125
    allocations, stddev = calc_min_variance_portfolio(return_vector, stddev_vector, correlation_matrix, target_return)
    print(f"scenario 1 - optimize portfolio for target return")
    print(f"target return: {target_return * 100.0:.2f}%")
    print(f"min variance portfolio: {allocations}")
    print(f"portfolio std deviation: {stddev * 100.0:.2f}%")

    print("-" * 40)

    target_variance = .15
    allocations, stddev, rtn = calc_max_return_portfolio(return_vector, stddev_vector, correlation_matrix, target_variance)
    print(f"scenario 2 - optimize portfolio for target variance")
    print(f"target variance: {target_variance * 100.0:.2f}%")
    print(f"max return: {allocations}")
    print(f"portfolio std deviation: {stddev * 100.0:.2f}%")
    print(f"portfolio return: {rtn * 100.0:.2f}%")
