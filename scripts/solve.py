
import matplotlib.pyplot as plt
import numpy as np

from phys702hw5 import solver

from math import pi


bounds = (0.0, pi/2.0)

y_init = 0

x_final = 3

NITER = 10


def get_xend(theta)->float:
    x,_,_ = solver.solve_python(y_init,theta,0.01)
    return x[-1]

def get_next_guess(theta_low,theta_high):
    return (theta_low+theta_high)/2


def is_too_high(xfinal:float):
    return xfinal > x_final

def get_new_bounds(theta_low,theta_high):
    theta = get_next_guess(theta_low,theta_high)
    xfinal = get_xend(theta)
    if is_too_high(xfinal):
        theta_low = theta
    else:
        theta_high = theta
    return (theta_low,theta_high)

for _ in range(NITER):
    (theta_low,theta_high) = get_new_bounds(*bounds)
    bounds = get_new_bounds(theta_low,theta_high)

print(f'Theta is between {bounds[0]*180/np.pi} and {bounds[1]*180/np.pi} degrees')

0
