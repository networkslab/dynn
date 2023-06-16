from torch import nn
from torch import Tensor

class CustomGELU(nn.GELU):
    def __init__(self):
        super(CustomGELU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return super(CustomGELU, self).forward(input)

    def forward_get_code(self, input: Tensor) -> (Tensor, Tensor):
        forward_out = self.forward(input)
        activation_code = forward_out > 0
        return forward_out, activation_code
