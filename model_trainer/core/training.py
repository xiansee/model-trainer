import lightning.pytorch as pl
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn, optim


class StepOutput(BaseModel):
    loss: Tensor
    true_output: Tensor
    model_output: Tensor

    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
   

class TrainingModule(pl.LightningModule):
    """
    Training module (based on Lightning) that initializes training and implements
    training, validation and test steps.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    loss_function : nn.Module
        Function to compute accuracy between true vs model output
    initial_lr : float, optional
        Initial learning rate, by default 0.01
    weight_decay : float, optional
        Optimizer weight decay, by default 0.001
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        initial_lr: float = 0.01,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay

    def training_step(
        self, 
        batch: list, 
        batch_idx: int
    ):
        """ Step for training dataset. """

        X, Y = batch
        Y_pred = self.model(X)

        training_loss = self.loss_fn(Y, Y_pred)
        self.log("training_loss", training_loss)

        return {
            "loss": training_loss,
            "step_output": StepOutput(
                loss=training_loss,
                true_output=Y_pred,
                model_output=Y_pred
            )
        }
    
    def validation_step(
        self, 
        batch: list, 
        batch_idx: int
    ):
        """ Step for validation dataset. """

        X, Y = batch
        Y_pred = self.model(X)

        validation_accuracy = self.loss_fn(Y, Y_pred)
        self.log("validation_accuracy", validation_accuracy)

        return {
            "loss": validation_accuracy,
            "step_output": StepOutput(
                loss=validation_accuracy,
                true_output=Y_pred,
                model_output=Y_pred
            )
        }
    
    def test_step(
        self, 
        batch: list, 
        batch_idx: int
    ):
        """ Step for test dataset. """

        X, Y = batch
        Y_pred = self.model(X)

        test_accuracy = self.loss_fn(Y, Y_pred)
        self.log("test_accuracy", test_accuracy)

        return {
            "loss": test_accuracy,
            "step_output": StepOutput(
                loss=test_accuracy,
                true_output=Y_pred,
                model_output=Y_pred
            )
        }

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.initial_lr, 
            weight_decay=self.weight_decay
        )

        return optimizer
