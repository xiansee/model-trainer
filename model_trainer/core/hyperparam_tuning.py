from abc import ABC
from typing import Optional

from pydantic import BaseModel


class Hyperparameter(BaseModel, ABC):
    """
    Hyperparameter base class.

    Parameters
    ----------
    name : str
        Description for hyperparameter
    """

    name: str


class IntegerHyperparameter(Hyperparameter):
    """
    Hyperparameter for integer type.

    Parameters
    ----------
    low : int
        Lowest value in tuning range
    high : int
        Highest value in tuning range
    log : bool, optional
        Whether to tune within a logarithmic range, by default False
    """

    low: int
    high: int
    log: Optional[bool] = False


class FloatHyperparameter(Hyperparameter):
    """
    Hyperparameter for float type.

    Parameters
    ----------
    low : float
        Lowest value in tuning range
    high : float
        Highest value in tuning range
    log : bool, optional
        Whether to tune within a logarithmic range, by default False
    """

    low: float
    high: float
    log: Optional[bool] = False
