from typing import Any, Callable

import lightning as L
import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import Callback
from optuna.trial import Trial
from torch import Tensor

from model_trainer.core.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from model_trainer.core.logging import Logger
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


def initialize_trial(
    training_config: TrainingConfig, trial: Trial, callbacks: list[Callback] = []
) -> tuple[pl.Trainer, TrainingModule, L.LightningDataModule]:
    """
    Initialize hyperparameter tuning trial based on Optuna framework.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    trial : Trial
        Optuna study trial
    callbacks : list[Callback]
        List of callbacks for Lightning trainer

    Returns
    -------
    tuple[pl.Trainer, TrainingModule, L.LightningDataModule]
        Initialized modules for fitting
    """

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
        max_epochs=training_config.max_epochs,
        max_time={"minutes": training_config.max_time},
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    return trainer, training_module, data_module


def get_objective_function(training_config: TrainingConfig, logger: Logger) -> Callable:
    """
    Get objective function for Optuna study.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    logger : Logger
        Hyperparameter tuning logger

    Returns
    -------
    Callable
        Objective function
    """

    def objective(trial: Trial) -> Tensor:
        """Objective function for Optuna study."""

        with logger.start_trial_logs(trial=trial):
            trainer, training_module, data_module = initialize_trial(
                training_config=training_config, trial=trial, callbacks=[logger]
            )
            trainer.fit(training_module, data_module)
            validation_accuracy = trainer.logged_metrics.get("validation_accuracy")

        return validation_accuracy

    return objective


def test_best_trial(
    training_config: TrainingConfig, logger: Logger, best_trial: Trial
) -> None:
    """
    Fit and test best trial from hyperparameter tuning study.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    logger : Logger
        Hyperparameter tuning logger
    best_trial : Trial
        Optuna's best trial
    """

    with logger.start_best_trial_logs(trial=best_trial):
        trainer, training_module, data_module = initialize_trial(
            training_config=training_config, trial=best_trial, callbacks=[logger]
        )
        trainer.fit(training_module, data_module)
        logger.log_model(model=trainer.model)

        trainer.test(training_module, data_module)

    return


def run_hyperparameter_tuning(training_config: TrainingConfig) -> None:
    """
    Start hyperparameter tuning using Optuna framework and log training using MLflow
    framework.

    Parameters
    ----------
    training_config : TrainingConfig
        User defined training configuration
    """

    logger = Logger(
        experiment=training_config.experiment,
        run_name=training_config.run_name,
        experiment_tags=training_config.experiment_tags,
    )

    objective = get_objective_function(training_config=training_config, logger=logger)

    with logger.start_hyperparameter_tuning_logs():
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=20, interval_steps=5
            ),
        )
        study.optimize(objective, n_trials=training_config.num_trials)

        test_best_trial(
            training_config=training_config, logger=logger, best_trial=study.best_trial
        )

    return study
