import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim, dense_size=128) -> None:
        """Many-to-one attention mechanism for PyTorch

        Args:
            input_dim (int): Input feature dimension (last dimension in input tensor)
            dense_size (int, optional): [description]. Defaults to 128.
        """
        super().__init__()
        self.hidden2score_first_part = nn.Linear(input_dim, input_dim, bias=False)
        self.preact2attention = nn.Linear(2 * input_dim, dense_size, bias=False)
        self.softmax_layer = nn.Softmax(dim=1)
        self.tanh_activation = nn.Tanh()

    def forward(self, inputs):
        """[summary]

        Args:
            inputs (torch.Tensor): input tensor of shape [B, T, D]. B is batch size, T is sequence length and D is
            the same as input dimension in constructor.

        Returns:
            torch.Tensor: 2D tensor with shape [B, H] where H is dense_size chosen in constructor.
        """
        hidden_states = inputs
        score_first_part = self.hidden2score_first_part(hidden_states)
        h_t = hidden_states[:, -1, :]
        score = torch.einsum("btd,bd->bt", (score_first_part, h_t))
        attention_weights = self.softmax_layer(score)
        context_vector = torch.einsum("btd,bt->bd", (hidden_states, attention_weights))
        pre_activation = torch.cat([context_vector, h_t], dim=1)
        attention_vector = self.preact2attention(pre_activation)
        attention_vector = self.tanh_activation(attention_vector)
        return attention_vector
