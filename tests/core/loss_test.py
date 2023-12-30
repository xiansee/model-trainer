import numpy as np
import pytest
import torch

from model_trainer.core.loss import MAE, MSE, RMSE, LossFunction


def test_loss_function_base_class():
    """Test LossFunction base class."""

    with pytest.raises(TypeError):
        # Missing abstract method
        class FooLoss(LossFunction):
            pass

        FooLoss()

    try:
        # Correct class definition
        class FooLoss(LossFunction):
            def forward(self, y_pred, y_true):
                pass

        FooLoss()

    except TypeError:
        pytest.fail(
            "LossFunction child class failed to instantiate with correct set of inputs."
        )


def test_loss_functions():
    """Test common loss functions."""

    Y1 = torch.randn(10)
    Y2 = torch.randn(10)

    MAE_loss = MAE()
    MSE_loss = MSE()
    RMSE_loss = RMSE()

    assert MAE_loss(Y1, Y2) == np.absolute(np.subtract(Y1, Y2)).mean()
    assert MSE_loss(Y1, Y2) == np.square(np.subtract(Y1, Y2)).mean()
    assert RMSE_loss(Y1, Y2) == np.sqrt(np.square(np.subtract(Y1, Y2)).mean())
