## model_trainer

`model_trainer` is a machine learning model training app. It supports automated hyperparameter tuning and training progress visualization. The following dependencies are used to build this app:
- `pytorch` and `lightning` for training, validation and test steps
- `optuna` for automated hyperparameters tuning
- `mlflow` for tracking training progress and visualizing hyperparameter tuning results

## Installation

You will require Docker to run the app. Docker can be installed [here](https://docs.docker.com/engine/install/).

Next download the Dockerfile from the [app](/app/) folder and build the container using the following command with Dockerfile in the same directory:
```
docker build .
```

To use it without Docker (i.e., not as an application), download the tarball from the latest release assets and install it through pip:
```
pip install model-trainer-1.0.0.tar.gz
```

## Usage

1. Copy the `docker-compose.yml` and `main.py` files from the [example](/example) folder.
2. To train a model, you will need to prepare a `data_module.py` file and `model.py` file, each containing the dataset and model that you would like to train. For example, here is a `model.py` file with a simple LSTM model:
```python
from torch import Tensor, nn


class LSTM(nn.Module):
    """Mock LSTM model with a fully connected layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_lstm_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X: Tensor) -> Tensor:
        lstm_output, _ = self.lstm(X)
        model_output = self.fc_layer(lstm_output)

        return model_output
```
3. Prepare a config.yaml that specifies the inputs to the dataset and model classes for training. Any input can be defined as a hyperparameter rather than a fixed value. For example, here is a config.yaml file to train the above LSTM model:
```yaml
experiment: foo_experiment
num_trials: 15
max_epochs: 20

model: # Specify arguments to intiialize model class
  input_size: 2
  hidden_size:
    hyperparameter_type: integer
    name: hidden_size
    low: 10
    high: 50
  output_size: 1
  num_lstm_layers:
    hyperparameter_type: integer
    name: num_lstm_layers
    low: 1
    high: 3

data_module: # Specify arguments to initialize data module class
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

optimizer:
  optimizer_algorithm: adam
  lr:
    hyperparameter_type: float
    name: learning_rate
    low: 0.001
    high: 0.1
    log: True

trainer:
  loss_function: rmse
```
4. Run the training app using:
```
docker-compose up -d
```
5. See the training logs using:
```
docker-compose logs model_trainer
```
6. Visualize hyperparameter tuning results by visiting the mlFlow app locally at `http://localhost:8080`.