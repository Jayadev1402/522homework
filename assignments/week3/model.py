import torch
from typing import Callable
import torch.nn as nn


class MLP(nn.Module):
    """
    This is the MLP implementation
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = nn.ReLU,
        initializer: Callable = nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        """
        this is the __init__
        """
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        """
        foward pass
        """
        x = self.dropout(torch.nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.bn2(self.fc2(x)))
        return x
