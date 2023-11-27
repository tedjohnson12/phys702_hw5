"""
Main module for polysolver

The goal of this module is to solve the
Lane-Emden equation:

.. math::
    \\frac{1}{\\xi^2} \\frac{d}{d\\xi}
    \\left( \\xi^2 \\frac{d\\theta_n}{d\\xi} \\right)
    = -\\theta_n^n

Now let's recast this with:

.. math::
    x = \\xi \\\\
    y = \\theta_n \\\\
    z = \\frac{d\\theta_n}{d\\xi} = \\frac{dy}{dx}

We are left with:

.. math::
    y' = \\frac{dy}{dx} = z \\\\
    z' = \\frac{dz}{dx} = -y^n - \\frac{2}{x} z

We will use a fourth-order Runge-Kutta method.

"""
from typing import Tuple, List

from phys702hw5 import runge_kutta
from phys702hw5 import derivatives


def solve_python(x_init:float,theta:float,h:float,max_iter=1000)->Tuple[List,List]:
    """
    Solve the Lane-Emden equation using a fourth-order Runge-Kutta method.
    
    Parameters
    ----------
    x_init : float
        The initial x value.
    theta : float
        The angle from the vertical.
    h : float
        The step size.
    max_iter : int, optional
        The maximum number of iterations. The default is 1000.
    """
    x_prev = x_init
    y_prev = 1
    z_prev = 0
    yprime = derivatives.get_yprime()
    zprime = derivatives.get_zprime(theta)
    n_iter = 0
    xs = []
    ys = []
    zs = []
    def stop_condition(x,y,z):
        if x>3.0:
            return True
        return False
        
    while not stop_condition(x_prev,y_prev,z_prev) and n_iter<max_iter:
        n_iter += 1
        xs.append(x_prev)
        ys.append(y_prev)
        zs.append(z_prev)
        x_next, y_next, z_next = runge_kutta.get_next_xyz(
            yprime,
            zprime,
            x_prev,
            y_prev,
            z_prev,
            h
        )
        x_prev, y_prev, z_prev = x_next, y_next, z_next
    return xs, ys, zs