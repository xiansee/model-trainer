import lightning as L
import pytest
from pydantic import ValidationError
from torch import nn

from model_trainer.core.hyperparameter import FloatHyperparameter, IntegerHyperparameter
from model_trainer.core.loss import RMSE
from model_trainer.core.training_config import (
    DataModuleConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
    process_user_config,
)
from tests.mock_data.data_module import DataModule
from tests.mock_data.dataset import RCCircuitDataset
from tests.mock_data.model import LSTM


def test_model_config():
    """Test initialization of TrainerConfig"""

    with pytest.raises(ValidationError):
        # Missing required input "model"
        ModelConfig(FooParam=1)

    try:
        # Correct input
        class Model(nn.Module):
            pass

        ModelConfig(model=Model())

    except ValidationError:
        pytest.fail("ModelConfig failed to initialize with correct set of inputs.")


def test_data_module_config():
    """Test initialization of DataModuleConfig"""

    with pytest.raises(ValidationError):
        # Missing required input "data_module"
        DataModuleConfig(FooParam=1)

        # Incorrect data_module type
        DataModuleConfig(data_module="data_module")

    try:
        # Correct input
        class DataModule(L.LightningDataModule):
            pass

        DataModuleConfig(data_module=DataModule())

    except ValidationError:
        pytest.fail("DataModuleConfig failed to initialize with correct set of inputs.")


def test_trainer_config():
    """Test initialization of TrainerConfig"""

    with pytest.raises(ValidationError):
        # Missing required input "loss_function"
        TrainerConfig(FooParam=1)

    try:
        # Correct input
        TrainerConfig(loss_function="rmse")

    except ValidationError:
        pytest.fail("TrainerConfig failed to initialize with correct set of inputs.")


def test_optimizer_config():
    """Test initialization of OptimizerConfig."""

    with pytest.raises(ValidationError):
        # Missing required input "optimizer_algorithm"
        OptimizerConfig(lr=0.1)

    try:
        # Correct inputs
        OptimizerConfig(optimizer_algorithm="adam", lr=0.1)

    except ValidationError:
        pytest.fail("OptimizerConfig failed to initialize with correct set of inputs.")


def test_training_config():
    """Test initialization of TrainingConfig"""

    class MockCorrectModel(nn.Module):
        pass

    class MockCorrectDataModule(L.LightningDataModule):
        pass

    mock_config = {
        "model": ModelConfig(model=MockCorrectModel()),
        "data_module": DataModuleConfig(data_module=MockCorrectDataModule()),
        "trainer": TrainerConfig(loss_function="rmse"),
        "optimizer": OptimizerConfig(optimizer_algorithm="adam", lr=0.01),
    }

    with pytest.raises(ValidationError):
        # Missing required fields
        TrainingConfig(experiment="FooExp")

        # Incorrect experiment type
        TrainingConfig(experiment=1, num_trials=1, max_epochs=10, **mock_config)

        # Incorrect num_trials type
        TrainingConfig(
            experiment="FooExp", num_trials=1.2, max_epochs=10, **mock_config
        )

        # Incorrect experiment_tags type
        TrainingConfig(
            experiment="FooExp",
            num_trials=1,
            max_epochs=10,
            experiment_tags={1: "FooTagValue"},
            **mock_config,
        )

        # Initialize with unsupported argument
        TrainingConfig(
            experiment="FooExp",
            num_trials=1,
            max_epochs=10,
            unsupported_arg="FooArg",
            **mock_config,
        )

    # Correct initialization
    try:
        TrainingConfig(experiment="FooExp", num_trials=1, max_epochs=10, **mock_config)

    except ValidationError:
        pytest.fail("TrainingConfig failed to validate a correct set of inputs.")


def test_process_user_config():
    """Test processing of user config into TrainingConfig."""

    mock_dataset = RCCircuitDataset(R=0.1, C=10, N_time_steps=1000, N_time_series=10)
    mock_model = LSTM

    mock_user_config = {
        "experiment": "foo_experiment",
        "num_trials": 10,
        "max_epochs": 20,
        "model": {
            "model": mock_model,
            "input_size": 2,
            "hidden_size": 10,
            "output_size": 1,
            "num_lstm_layers": {
                "hyperparameter_type": "integer",
                "name": "num_lstm_layers",
                "low": 1,
                "high": 3,
            },
        },
        "data_module": {"data_module": DataModule, "dataset": mock_dataset},
        "optimizer": {
            "optimizer_algorithm": "adam",
            "lr": {
                "hyperparameter_type": "float",
                "name": "learning_rate",
                "low": 0.001,
                "high": 0.1,
                "log": True,
            },
        },
        "trainer": {"loss_function": "rmse"},
    }

    training_config = process_user_config(mock_user_config)

    assert isinstance(training_config, TrainingConfig)
    assert isinstance(training_config.model, ModelConfig)
    assert isinstance(training_config.data_module, DataModuleConfig)
    assert isinstance(training_config.optimizer, OptimizerConfig)
    assert isinstance(training_config.trainer, TrainerConfig)

    assert training_config.experiment == "foo_experiment"
    assert training_config.num_trials == 10
    assert training_config.max_epochs == 20
    assert training_config.data_module.data_module == DataModule
    assert training_config.data_module.dataset == mock_dataset
    assert isinstance(training_config.optimizer.lr, FloatHyperparameter)
    assert isinstance(training_config.trainer.loss_function, RMSE)

    assert training_config.model.model == mock_model
    assert training_config.model.output_size == 1
    assert isinstance(training_config.model.num_lstm_layers, IntegerHyperparameter)
