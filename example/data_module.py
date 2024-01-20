import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


def calculate_V(
    t_series: list[float],
    i_series: list[float],
    R: float = 1,
    C: float = 10,
    i_R0: float = 0,
) -> list[float]:
    """
    Calculates the voltage response of a single parallel resistor-capacitor circuit.

    Parameters
    ----------
    t_series : list[float]
        Time data in seconds
    i_series : list[float]
        Time series current in Amperes
    R : float, optional
        Resistance in ohms, by default 1
    C : float, optional
        Capacitance in Farads, by default 10
    i_R0 : float, optional
        Initial current through resistor in Amperes, by default 0

    Returns
    -------
    list[float]
        Time series voltage in Volts
    """

    V_series = [i_R0 * R]
    delta_t_series = np.array(t_series[1:]) - np.array(t_series[:-1])

    for delta_t, i in zip(delta_t_series, i_series[1:]):
        # Discrete time equation for parallel R-C circuit
        i_R = np.exp(-delta_t / (R * C)) * i_R0 + (1 - np.exp(-delta_t / (R * C))) * i

        V_series.append(i_R * R)
        i_R0 = i_R

    return V_series


class RCCircuitDataset(Dataset):
    """
    Mock dataset for the time series voltage response of a single parallel resistor
    capacitor circuit.

    Parameters
    ----------
    R : float, optional
        Resistance in ohms, by default 1
    C : float, optional
        Capacitance in Farads, by default 10
    N_time_steps : int
        Number of time steps for each time series
    N_time_series : int
        Number of time series
    """

    def __init__(
        self,
        R: float = 1,
        C: float = 10,
        N_time_steps: int = 300,
        N_time_series: int = 3,
    ) -> None:
        if N_time_series < 3:
            raise ValueError(
                "RCCircuitDataset should be initialized with at least 3 datasets \
                for train-validation-test split."
            )

        self.R = R
        self.C = C
        self.N_time_steps = N_time_steps
        self.N_time_series = N_time_series
        self.data = []
        self.generate_mock_data()

    def generate_mock_data(self) -> None:
        """Generate mock data with randomized time series current."""

        t_data = [np.arange(0, self.N_time_steps, 1) for _ in range(self.N_time_series)]
        i_data = [np.random.rand(len(t)) for t in t_data]

        V_data = [
            calculate_V(t_series=t, i_series=i, R=self.R, C=self.C)
            for t, i in zip(t_data, i_data)
        ]

        self.data = [[i, V] for i, V in zip(i_data, V_data)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor]:
        data = self.data[index]
        data_length = len(data[0])

        X = torch.stack([torch.tensor(data[0], dtype=torch.float32)], dim=1)
        Y = torch.tensor(data[-1], dtype=torch.float32).view(data_length, 1)

        return X, Y


class DataModule(L.LightningDataModule):
    """Data module that splits dataset into train, validation and test."""

    def __init__(
        self,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ) -> None:
        super().__init__()

        if round(sum([train_split, val_split, test_split]), 6) != 1:
            raise ValueError("All of train/val/test splits must sum up to 1.")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.dataset = RCCircuitDataset(
            R=0.1, C=10, N_time_steps=1000, N_time_series=10
        )

    def setup(self, stage: str) -> None:
        num_dataset = len(self.dataset)
        num_training_set = round(self.train_split * num_dataset)
        num_validation_set = round(self.val_split * num_dataset)
        num_test_set = round(self.test_split * num_dataset)

        self.training_set, self.validation_set, self.test_set = random_split(
            self.dataset, [num_training_set, num_validation_set, num_test_set]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=1, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_set, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=1)
