from abc import ABC
from typing import Optional, Union

from pydantic import BaseModel


class Hyperparameter(BaseModel, ABC):
    """
    Hyperparameter base class.

    Parameters
    ----------
    name : str
        Name of hyperparameter
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
        Whether to sample from log domain, by default False
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
        Whether to sample from log domain, by default False
    """

    low: float
    high: float
    log: Optional[bool] = False


class CategoricalHyperparameter(Hyperparameter):
    """
    Hyperparameter for float type.

    Parameters
    ----------
    choices : bool | int | float | str
        Categorical choices
    """

    choices: list[Union[bool, int, float, str]]
