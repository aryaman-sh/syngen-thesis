import torch.nn as nn
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class activation(nn.Module):
    """ Applies an adapted Gaussian distribution function to each channel of an input 
    Tensor.

    Adaptated Gaussian distribution function is defined as:

    .. math::
        \text{activation}(x)  = \exp{\frac{-x**2}{sigma**2}}

    Shape: 
        - Input: Tensor of shape (n_batchs, n_channels, dims)
        - Output: same shape as the input
    
    Args:
        num_parameters (int): number of channels
    Returns:
        a Tensor of the same dimension and shape as the input
    
    """
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super(activation, self).__init__()
        self.sigma = Parameter(torch.reshape(torch.empty(
            num_parameters, **factory_kwargs).fill_(init), (1, num_parameters, 1, 1, 1)))

    def forward(self, input: Tensor) -> Tensor:
        x = -input**2/self.sigma**2
        x = torch.exp(x)
        return x

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 1, 3, padding=1)
        self.act1 = activation(16)
        self.act2 = activation(16)

    def forward(self, input):
        x = self.conv1(input)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        return x

class Normalizer_Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv3d(16, 1, 3, padding=1)
        self.act1 = activation(16)
        self.act2 = activation(32)
        self.act3 = activation(16)

    def forward(self, input):
        x = self.conv1(input)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)

        return x
    
class Normalizer_Deeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv3d(16, 1, 3, padding=1)
        self.act1 = activation(16)
        self.act2 = activation(32)
        self.act3 = activation(32)
        self.act4 = activation(16)

    def forward(self, input):
        x = self.conv1(input)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)

        return x