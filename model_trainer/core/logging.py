import math
import time
from contextlib import contextmanager
from datetime import datetime

import lightning.pytorch as pl
import mlflow
from lightning.pytorch.callbacks import Callback


class Logger(Callback):
    """_summary_

    Parameters
    ----------
    Callback : _type_
        _description_
    """

    def __init__(
        self, experiment: str, run_name: str = None, experiment_tags: dict = {}
    ):
        self._experiment = experiment
        self._run_name = run_name
        self._experiment_tags = experiment_tags

        self.init_time = datetime.utcnow().strftime("%Y_%m_%d_T%H_%M_%SZ")
        self.fit_start_timestamp = float("nan")
        self.epoch_start_timestamp = float("nan")

    @property
    def experiment(self) -> str:
        """MLflow experiment name."""

        return self._experiment

    @property
    def experiment_id(self) -> int:
        """ID of MLflow experiment."""

        if experiment := mlflow.get_experiment_by_name(self.experiment):
            return experiment.experiment_id

        return mlflow.create_experiment(self.experiment)

    @property
    def run_name(self) -> str:
        """Name of MLflow run, defaults to a timestamp."""

        if self._run_name is not None:
            return self._run_name

        return f"{self.init_time}"

    @property
    def experiment_tags(self) -> dict:
        """MLflow experiment tags."""

        return self._experiment_tags

    @contextmanager
    def start_hyperparameter_tuning_logs(self) -> None:
        try:
            yield mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                tags=self.experiment_tags,
                description=f"Hyperparameter tuning for run {self.run_name}",
            )
        finally:
            mlflow.end_run()

    @contextmanager
    def start_trial_logs(self, trial_number: int, run_name: str = None) -> None:
        try:
            yield mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"trial_{trial_number}" if run_name is None else run_name,
                description=f"Trial #{trial_number} for run {self.run_name}",
                nested=True,
            )
        finally:
            mlflow.end_run()

    def get_current_timestamp(self) -> int:
        """
        Get current timestamp in miliseconds.

        Returns
        -------
        int
            Unix timestamp in milliseconds.
        """
        return round(time.time() * 1000)

    def get_time_elapsed_for(self, operation: str) -> int:
        """
        Get time elapsed for a particular operation (e.g., fit, epoch).

        Parameters
        ----------
        operation : str
            Operation type (e.g., fit, epoch)

        Returns
        -------
        int
            Time elapsed in milliseconds.

        Raises
        ------
        ValueError
            If operation is not supported.
        """
        options = ["fit", "epoch"]
        operation = operation.lower()
        if operation not in options:
            raise ValueError(
                f"Incorrect operation to calculate timedelta, please choose from the following: {options}"
            )

        start_timestamp = getattr(self, f"{operation}_start_timestamp")
        if math.isnan(start_timestamp):
            return float("nan")
        else:
            return round((time.time() - start_timestamp) * 1000)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """PyTorch module callback for training start."""

        self.fit_start_timestamp = time.time()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """PyTorch module callback for training epoch start."""

        self.epoch_start_timestamp = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """PyTorch module callback for training epoch end."""

        epoch_num = pl_module.current_epoch
        training_loss = float(trainer.logged_metrics.get("training_loss", "nan"))

        mlflow.log_metric(
            key="training_loss",
            value=training_loss,
            step=epoch_num,
            timestamp=self.get_current_timestamp(),
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """PyTorch module callback for validation epoch end."""

        # TODO: skip epoch_num == 0??

        epoch_num = pl_module.current_epoch
        epoch_time = self.get_time_elapsed_for("epoch")
        validation_accuracy = float(
            trainer.logged_metrics.get("validation_accuracy", "nan")
        )

        mlflow.log_metric(
            key="validation_accuracy",
            value=validation_accuracy,
            step=epoch_num,
            timestamp=self.get_current_timestamp(),
        )
        mlflow.log_metric(
            key="epoch_time",
            value=epoch_time,
            step=epoch_num,
            timestamp=self.get_current_timestamp(),
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """PyTorch module callback for fit end."""

        training_time = self.get_time_elapsed_for("fit")
        mlflow.log_metric(
            key="training_time",
            value=training_time,
            timestamp=self.get_current_timestamp(),
        )
