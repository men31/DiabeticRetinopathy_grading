from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def GlobalMaxPool2d(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(torch.amax(x, dim=2, keepdim=True), dim=3, keepdim=True)


def GlobalAvgPool2d(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=[2, 3], keepdim=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)
        if hidden_dim is None:
            self.linear_hidden_1 = nn.Linear(dim, 2 * dim)
            self.linear_hidden_1 = nn.Linear(2 * dim, dim)
        else:
            self.linear_hidden_1 = nn.Linear(dim, hidden_dim)
            self.linear_hidden_2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        output = self.linear_in(x)
        output = self.linear_hidden_1(output)
        swish = output * torch.sigmoid(output)
        swish = self.linear_hidden_2(swish)
        swiglu = swish * self.linear_out(x)

        return swiglu


class View(nn.Module):
    def __init__(self, shape: Union[tuple, int]):
        super(View, self).__init__()
        self.shape = shape

    def __repr__(self):
        return f"View{self.shape}"

    def forward(self, input):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = input.size(0)
        if type(self.shape) == tuple:
            shape = (batch_size, *self.shape)
        elif type(self.shape) == int:
            shape = (batch_size, self.shape)
        out = input.v


def load_state_from_ckpt(model: nn.Module, ckpt_dir:str):
    ckpt_dict = torch.load(ckpt_dir)
    model.load_state_dict(ckpt_dict['state_dict'])
    return model
