from typing import Optional

import lightning as L
from pydantic import AfterValidator, BaseModel
from torch import nn
from typing_extensions import Annotated

from model_trainer.core.hyperparam_tuning import Hyperparameter
from model_trainer.core.loss import get_loss_function
from model_trainer.core.optimizer import get_optimizer


class ModelConfig(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """
    Model configuration base model for data validation.

    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    """

    model: nn.Module


class DataModuleConfig(BaseModel, extra="allow", arbitrary_types_allowed=True):
    """
    Data module configuration base model for data validation.

    Parameters:
    -----------
    data_module : L.LightningDataModule
        Lightning data module
    """

    data_module: L.LightningDataModule


class TrainerConfig(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Trainer configuration base model for data validation.

    Parameters:
    -----------
    loss_function : str
        Name of loss function
    """

    loss_function: Annotated[str, AfterValidator(get_loss_function)]


class OptimizerConfig(BaseModel, extra="allow"):
    """
    Optimizer configuration base model for data validation.

    Parameters:
    -----------
    optimizer_algorithm : str
        Optimizer algorithm
    lr : float | Hyperparameter
        Learning rate
    """

    optimizer_algorithm: Annotated[str, AfterValidator(get_optimizer)]
    lr: float | Hyperparameter


class ExperimentConfig(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Experiment configuration base model for data validation.

    Parameters:
    -----------
    experiment : str
        Name of experiment
    num_trials: int
        Number of hyperparameter tuning trials
    experiment_tags : dict[str, str], Optional
        Tags for experiment, by default {}
    run_name : str, Optional
        Name of run, by default None
    artifact_path : str, Optional
        Path of artifact, by default None
    max_epochs : int, Optional
        Max number of epoch per training, by default None
    max_time : float, Optional
        Max time for each training in minutes, by default None
    model : ModelConfig
        Settings for model
    data_module : DataModuleConfig
        Settings for data module
    trainer : TrainerConfig
        Settings for trainer
    optimizer : OptimizerConfig
        Settings for optimizer
    """

    experiment: str
    num_trials: int = 10
    experiment_tags: Optional[dict[str, str]] = {}
    run_name: Optional[str] = None
    artifact_path: Optional[str] = None
    max_epochs: Optional[int] = None
    max_time: Optional[float] = None  # [minutes]

    model: ModelConfig
    data_module: DataModuleConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
