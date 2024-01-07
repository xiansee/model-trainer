from typing import Any, Callable

import lightning.pytorch as pl
import optuna
from optuna.trial import Trial
from torch import Tensor

from model_trainer.core.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from model_trainer.core.training import TrainingModule
from model_trainer.core.training_config import TrainingConfig


def define_module(trial: Trial, module: Any, module_init_params: dict) -> Any:
    """
    Instantiate module with user defined parameters. Handles hyperparameters per
    Optuna's documentations.

    Parameters
    ----------
    trial : Trial
        Optuna's trial object
    module : Any
        Module class
    module_init_params : dict
        Parameters used for module instantiation

    Returns
    -------
    Any
        Instantiated module
    """

    def is_hyperparameter(input_type: Any) -> bool:
        """Returns whether input is a hyperparameter."""

        return input_type in suggest_functions.keys()

    suggest_functions = {
        IntegerHyperparameter: trial.suggest_int,
        FloatHyperparameter: trial.suggest_float,
        CategoricalHyperparameter: trial.suggest_categorical,
    }
    processed_init_params = {}

    for module_arg, value in module_init_params.items():
        if value == module:
            continue

        value_type = type(value)
        if is_hyperparameter(value_type):
            suggest_fn = suggest_functions.get(value_type)
            processed_init_params[module_arg] = suggest_fn(**dict(value))

        else:
            processed_init_params[module_arg] = value

    return module(**processed_init_params)


def get_objective_function(training_config: TrainingConfig) -> Callable:
    """
    Get objective function for Optuna study.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration

    Returns
    -------
    Callable
        Objective function
    """

    def objective(trial: Trial) -> Tensor:
        """Objective function for Optuna study."""

        # Init model
        model_config = training_config.model
        model_class = model_config.model
        model_init_params = dict(model_config)
        model = define_module(
            trial=trial, module=model_class, module_init_params=model_init_params
        )

        # Init data module
        data_module_config = training_config.data_module
        data_module_class = data_module_config.data_module
        data_module_init_params = dict(data_module_config)
        data_module = define_module(
            trial=trial,
            module=data_module_class,
            module_init_params=data_module_init_params,
        )

        # Init optimizer
        optimizer_config = training_config.optimizer
        optimizer_class = optimizer_config.optimizer_algorithm
        optimizer_init_params = dict(optimizer_config)
        optimizer_init_params.update({"params": model.parameters()})
        optimizer = define_module(
            trial=trial,
            module=optimizer_class,
            module_init_params=optimizer_init_params,
        )

        # Init training module
        training_module = TrainingModule(
            model=model,
            optimizer=optimizer,
            loss_function=training_config.trainer.loss_function,
        )

        # Init trainer
        trainer = pl.Trainer(
            default_root_dir="../lightining_logs",
            max_epochs=training_config.max_epochs,
            max_time={"minutes": training_config.max_time},
            logger=False,
            callbacks=[],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

        trainer.fit(training_module, data_module)

        validation_accuracy = trainer.logged_metrics.get("validation_accuracy")
        return validation_accuracy

    return objective


def run_hyperparameter_tuning(training_config: TrainingConfig) -> None:
    """
    Start hyperparameter tuning using Optuna framework.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    """

    objective = get_objective_function(training_config=training_config)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=20, interval_steps=5
        ),
    )

    study.optimize(objective, n_trials=training_config.num_trials)
    return study
