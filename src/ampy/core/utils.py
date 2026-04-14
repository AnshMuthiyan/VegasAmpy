import math
import pathlib

import numpy as np

from ampy.core.structs import ScaleType


def sec_to_days(x):
    """ Convert seconds to days. """
    return x / 86400.0


def days_to_sec(x):
    """ Convert days to seconds. """
    return x * 86400.0


def to_scale(value, pre, post) -> float:
    """
    Converts `value` from `pre` to `post` scale.

    Parameters
    ----------
    value : float
        The value to convert.

    pre : str or ScaleType
        The scale of `value`.

    post : str or ScaleType
        The new scale type.

    Returns
    -------
    float
        The converted value.
    """
    if isinstance(pre, str):
        from_s = ScaleType(pre)

    if isinstance(post, str):
        post = ScaleType(post)

    if pre == post:
        return value

    match post:
        case ScaleType.LOG:
            return to_log(value, pre)
        case ScaleType.LN:
            return to_ln(value, pre)
        case ScaleType.LINEAR:
            return to_linear(value, pre)
        case _:
            raise NotImplementedError


def to_log(value, scale) -> float:
    """
    Converts `value` from `scale` to log10.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of `value`.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LOG:
            return value
        case ScaleType.LN:
            return value / math.log(10)
        case ScaleType.LINEAR:
            return math.log10(value)
        case _:
            raise NotImplementedError


def to_ln(value, scale) -> float:
    """
    Converts `value` from `scale` to log.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of `value`.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LN:
            return value
        case ScaleType.LOG:
            return value * math.log(10)
        case ScaleType.LINEAR:
            return math.log(value)
        case _:
            raise NotImplementedError


def to_linear(value: float, scale: ScaleType) -> float:
    """
    Converts `value` from `scale` to linear.

    Parameters
    ----------
    value : float
        The value to convert.

    scale : ScaleType
        The scale of `value`.

    Returns
    -------
    float
        The converted value.
    """
    match scale:
        case ScaleType.LINEAR:
            return value
        case ScaleType.LOG:
            return 10 ** value
        case ScaleType.LN:
            return math.exp(value)
        case _:
            raise NotImplementedError


def chi_squared(f, y, e, s = None) -> float:
    """
    Calculates the chi-squared value.

    Parameters
    ----------
    f : np.ndarray of float
        The predicted values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float or np.ndarray of float, optional
        The slop parameter.

    Returns
    -------
    float or Quantity
        The chi-squared value.
    """
    if s is None:
        return np.sum(((y - f) / e) ** 2)

    return chi_squared_eff(f, y, e, s)


def chi_squared_eff(f, y, e, s) -> float:
    """
    When the slop parameter, `s`, is provided, the chi-squared calculation
    accounts for additional unknown variances. When `s > 0`, the effective
    uncertainties increase, decreasing the penalty for model-data mismatches
    but adding a penalty for increasing `s` through the normalization term.

    Parameters
    ----------
    f : np.ndarray of float
        The modeled values.

    y : np.ndarray of float
        The observed values.

    e : np.ndarray of float
        The uncertainty in the observed values.

    s : float or np.ndarray of float
        The log slop parameter.

    Returns
    -------
    float
        The effective chi-squared value.
    """
    # Convert slop to linear space linear-space average slop
    s_lin_avg = f * (10**s - 10**-s) / 2

    # Combine the slop and data uncertainties
    sig = np.sqrt(s_lin_avg ** 2 + e ** 2)

    # return chi-squared effective
    return np.sum(2 * np.log(sig) + ((y - f) / sig) ** 2)


def hill(x1, x2, w):
    """ Decreasing hill function. """
    return w * x1 + (1.0 - w) * x2


def crosses(x, y):
    """
    Do ``x`` and ``y`` cross?

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of same shape.

    Returns
    -------
    int
        The first index of crossing. If no crossing was found, then returns -1.
    """
    cross_idx = np.where(np.diff(np.sign(x - y)))[0]
    return -1 if len(cross_idx) == 0 else cross_idx[0]
