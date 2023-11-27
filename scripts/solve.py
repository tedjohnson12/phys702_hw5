
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from phys702hw5 import solver

from math import pi


bounds = (0.0, pi/2.0)

YINIT = 0

XFINAL = 3

NITER = 10
STEP_SIZE = 0.01
FMT = '.4f'

def get_xend(theta)->float:
    """
    Get the final x value for a given angle.
    
    Parameters
    ----------
    theta : float
        The angle in radians.
    
    Returns
    -------
    float
        The final x value.
    """
    x,_,_ = solver.solve_python(YINIT,theta,STEP_SIZE)
    return x[-1]

def get_next_guess(theta_low,theta_high):
    """
    Get next guess.
    
    Parameters
    ----------
    theta_low : float
        The lower bound on theta.
    theta_high : float
        The upper bound on theta.
    
    Returns
    -------
    float
        The next theta to guess
    
    """
    return (theta_low+theta_high)/2

def is_too_high(xfinal:float):
    """
    If the final x value is too high.
    
    Parameters
    ----------
    xfinal : float
        The final x value.
    
    Returns
    -------
    bool
        If the final x value is too high.
    """
    return xfinal > XFINAL

def get_new_bounds(theta_low,theta_high):
    """
    Update the bounds.
    
    Parameters
    ----------
    theta_low : float
        The lower bound on theta.
    theta_high : float
        The upper bound on theta.
    
    Returns
    -------
    tuple
        The new bounds.
    """
    theta = get_next_guess(theta_low,theta_high)
    xfinal = get_xend(theta)
    if is_too_high(xfinal):
        theta_low = theta
    else:
        theta_high = theta
    return (theta_low,theta_high)

if __name__ in ['__main__']:
    for _ in range(NITER):
        (theta1,theta2) = get_new_bounds(*bounds)
        bounds = get_new_bounds(theta1,theta2)
    
    print(f'Theta is between {bounds[0]*180/np.pi:{FMT}} and {bounds[1]*180/np.pi:{FMT}} degrees')
    
    theta = get_next_guess(*bounds)
    
    x,y,z = solver.solve_python(YINIT,theta,STEP_SIZE)
    
    plt.plot(x,y,label='$\\phi$ (rad)')
    plt.plot(x,z,label='$\\omega$ (rad/s)')
    
    plt.axvline(x=XFINAL,color='k',linestyle='dashed')
    plt.axhline(y=pi/2,color='k',linestyle='dashed')
    
    plt.xlabel('time (s)')
    plt.title(f'$\\theta$ = {theta*180/np.pi:{FMT}} degrees')
    plt.legend()
    
    plt.savefig(Path(__file__).parent / 'solution.png')
    
