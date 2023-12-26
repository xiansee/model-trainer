import numpy as np
import torch

from model_trainer.core.loss import MAE, MSE, RMSE


def test_loss_functions():
    """Test loss functions for model training."""

    Y1 = torch.randn(10)
    Y2 = torch.randn(10)

    MAE_loss = MAE()
    MSE_loss = MSE()
    RMSE_loss = RMSE()

    assert MAE_loss(Y1, Y2) == np.absolute(np.subtract(Y1, Y2)).mean()
    assert MSE_loss(Y1, Y2) == np.square(np.subtract(Y1, Y2)).mean()
    assert RMSE_loss(Y1, Y2) == np.sqrt(np.square(np.subtract(Y1, Y2)).mean())
