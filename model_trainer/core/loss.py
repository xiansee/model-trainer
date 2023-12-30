from abc import ABC, abstractmethod

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
