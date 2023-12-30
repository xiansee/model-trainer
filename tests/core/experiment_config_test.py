import lightning as L
import pytest
from pydantic import ValidationError
from torch import nn

from model_trainer.core.experiment_config import (
    DataModuleConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)


def test_model_config():
    """Test initialization of TrainerConfig"""

    with pytest.raises(ValidationError):
        # Missing required input "model"
        TrainerConfig(FooParam=1)

        # Incorrect model type
        ModelConfig(model="model")

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


def test_experiment_config():
    """Test initialization of ExperimentConfig"""

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
        ExperimentConfig(experiment="FooExp")

        # Incorrect experiment type
        ExperimentConfig(experiment=1, num_trials=1, **mock_config)

        # Incorrect num_trials type
        ExperimentConfig(experiment="FooExp", num_trials=1.2, **mock_config)

        # Incorrect experiment_tags type
        ExperimentConfig(
            experiment="FooExp",
            num_trials=1,
            experiment_tags={1: "FooTagValue"},
            **mock_config,
        )

        # Initialize with unsupported argument
        ExperimentConfig(
            experiment="FooExp", num_trials=1, unsupported_arg="FooArg", **mock_config
        )

    # Correct initialization
    try:
        ExperimentConfig(experiment="FooExp", num_trials=1, **mock_config)

    except ValidationError:
        pytest.fail("ExperimentConfig failed to validate a correct set of inputs.")
