import lightning.pytorch as pl

from model_trainer.core.loss import RMSE
from model_trainer.core.optimizer import OptimizerConfig, get_optimizer
from model_trainer.core.training import TrainingModule
from tests.mock_data.data_module import DataModule
from tests.mock_data.dataset import RCCircuitDataset
from tests.mock_data.model import LSTM


def test_training_module():
    """Test for training module."""

    mock_dataset = RCCircuitDataset(R=0.1, C=10, N_time_steps=1000, N_time_series=10)
    data_module = DataModule(dataset=mock_dataset)
    model = LSTM(
        input_size=1,
        hidden_size=10,
        output_size=1,
        num_lstm_layers=1,
    )
    loss_fn = RMSE()
    optimizer = get_optimizer(name="adam")
    optimizer_config = OptimizerConfig(lr=0.01, weight_decay=0.001)

    training_module = TrainingModule(
        model=model,
        loss_function=loss_fn,
        optimizer_type=optimizer,
        optimizer_config=optimizer_config,
    )

    assert training_module.model == model
    assert training_module.loss_fn == loss_fn
    assert getattr(training_module, "training_step")
    assert getattr(training_module, "validation_step")

    trainer = pl.Trainer(
        max_epochs=50,
        max_time={"seconds": 10},
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    trainer.fit(training_module, data_module)
    assert trainer.logged_metrics.get("validation_accuracy") < 0.015

    trainer.test(training_module, data_module)
    assert trainer.logged_metrics.get("test_accuracy") < 0.015
