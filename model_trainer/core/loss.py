import torch
from torch import nn


class RMSE(nn.Module):
    """Root mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse_loss_fn = nn.MSELoss()
        rmse = torch.sqrt(mse_loss_fn(y_pred, y_true))
        return rmse


class MAE(nn.Module):
    """Mean absolute error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


class MSE(nn.Module):
    """Mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        mse_loss_fn = nn.MSELoss()
        return mse_loss_fn(y_pred, y_true)
