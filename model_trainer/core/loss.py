from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn


class LossFunction(nn.Module, ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class RMSE(LossFunction):
    """Root mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        rmse = torch.sqrt(mse_loss_fn(y_pred, y_true))
        return rmse


class MAE(LossFunction):
    """Mean absolute error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean(torch.abs(y_pred - y_true))


class MSE(LossFunction):
    """Mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        return mse_loss_fn(y_pred, y_true)


class _LossFunctionChoices(str, Enum):
    """Supported loss function options used for data validation."""

    rmse: str = "rmse"
    mae: str = "mae"
    mse: str = "mse"


def get_loss_function(name: str) -> LossFunction:
    """
    Get loss fcuntion class.

    Parameters
    ----------
    name : str
        Name of loss function

    Returns
    -------
    LossFunction
        Loss function class

    Raises
    ------
    ValueError
        If the name of loss function is not an invalid option.
    NotImplementedError
        If the loss function is not implemented. This is a development error.
    """

    match name.lower():
        case "rmse":
            return RMSE()

        case "mae":
            return MAE()

        case "mse":
            return MSE()

        case _:
            _supported_choices = [option.value for option in _LossFunctionChoices]
            if name not in _supported_choices:
                raise ValueError(
                    f"{name} loss function not supported. Please select from {_supported_choices}"
                )

            raise NotImplementedError(
                f"{name} loss function not implemented by developer."
            )
