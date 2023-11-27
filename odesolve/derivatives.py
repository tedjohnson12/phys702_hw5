"""

Derivates to use in the Runge-Kutta method.

"""
from typing import Callable
from math import sin,cos

G = 9.8 # m s-2
A = 0.45 # m


def get_yprime() -> Callable:
    """
    Get the yprime function.

    Returns
    -------
    Callable
        The yprime function
    """
    def yprime(x, y, z):
        """
        :math:`\\frac{dy}{dx}`

        Parameters
        ----------
        x : float
            The x value.
        y : float
            The y value.
        z : float
            The z value.

        Returns
        -------
        float
            The yprime value.
        """
        return z
    return yprime


def get_zprime(theta:float) -> Callable:
    """
    Get the zprime function.

    Parameters
    ----------
    theta: float
        The angle from the vertical in radians

    Returns
    -------
    Callable
        The zprime function
    """
    def zprime(x, y, z):
        """
        :math:`\\frac{dz}{dx}`

        Parameters
        ----------
        x : float
            The x value.
        y : float
            The y value.
        z : float
            The z value.

        Returns
        -------
        float
            The zprime value.
        """
        return (3*G)/(2*A) * sin(theta) * cos(y)
    return zprime
