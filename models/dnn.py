import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkAttackDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.activation = activation.lower()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        if self.activation == "tanh":
            return nn.Tanh()
        if self.activation == "sigmoid":
            return nn.Sigmoid()
        return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dnn(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    use_residual: bool = False,
    activation: str = "relu",
) -> NetworkAttackDNN:
    return NetworkAttackDNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual,
        activation=activation,
    )
