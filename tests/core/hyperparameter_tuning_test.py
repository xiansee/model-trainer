from model_trainer.core.hyperparameter import FloatHyperparameter, IntegerHyperparameter
from model_trainer.core.hyperparameter_tuning import (
    define_module,
    run_hyperparameter_tuning,
)
from model_trainer.core.training_config import ModelConfig, process_user_config
from tests.mock_data.data_module import DataModule
from tests.mock_data.dataset import RCCircuitDataset
from tests.mock_data.model import LSTM


def test_define_module():
    """Test define module and correct handling of hyperparameters."""

    class MockModel:
        def __init__(self, arg_1: int, arg_2: int, arg_3: float):
            self.arg_1 = arg_1
            self.arg_2 = arg_2
            self.arg_3 = arg_3

    class MockTrial:
        def suggest_int(self, **kwargs):
            return f"suggest_int: {kwargs}"

        def suggest_float(self, **kwargs):
            return f"suggest_float: {kwargs}"

        def suggest_categorical(self, **kwargs):
            return f"suggest_categorical: {kwargs}"

    model_settings = {
        "model": MockModel,
        "arg_1": 1,
        "arg_2": IntegerHyperparameter(name="arg_2", low=1, high=3),
        "arg_3": FloatHyperparameter(name="arg_3", low=1.5, high=5.5),
    }
    model_config = ModelConfig(**model_settings)
    model_init_params = dict(model_config)
    module = define_module(
        trial=MockTrial(),
        module=model_config.model,
        module_init_params=model_init_params,
    )

    assert module.arg_1 == 1
    assert "suggest_int" in module.arg_2
    assert "suggest_float" in module.arg_3


def test_hyperparameter_tuning():
    """Test execution and completion of hyperparameter tuning."""

    mock_dataset = RCCircuitDataset(R=0.1, C=10, N_time_steps=1000, N_time_series=10)
    mock_model = LSTM

    mock_user_config = {
        "experiment": "foo_experiment",
        "num_trials": 3,
        "max_epochs": 20,
        "max_time": 0.1,
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
    study = run_hyperparameter_tuning(training_config=training_config)

    assert len(study.trials) == 3
    assert set(study.best_params.keys()) == set(["learning_rate", "num_lstm_layers"])
