import torch
import torch.nn             as nn 
# import torch.nn.functional  as F

class Attention(nn.Module):
    """Attention mechanism.

    Parameters:
    - dim       (int)   : The input and output dimension of per-token features.
    - n_heads   (int)   : Number of attention heads.
    - qkv_bias  (bool)  : If True, include bias in the query, key, and value projections.
    - attn_p    (float) : Dropout probability applied to the query, key, and value tensors.
    - proj_p    (float) : Dropout probability applied to the output tensor.

    Attributes:
    - scale                 (float)     : Normalizing constant for the dot product.
    - qkv                   (nn.Linear) : Linear projection for the query, key, and value.
    - attn_drop, proj_drop  (nn.Dropout): Dropout layers.
    - proj                  (nn.Linear) : Linear mapping that takes the concatenated output of all 
    attention heads and maps it into a new space.
    """
    def __init__(self, 
                 dim,
                 n_heads=8,
                 qkv_bias=True,
                 attn_p=0.,
                 proj_p=0.)-> None:
        
        super().__init__()
        self.n_heads    = n_heads
        self.dim        = dim
        self.head_dim   = dim // n_heads
        self.scale      = self.head_dim ** -0.5

        # Linear projections for query, key, and value
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_p)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_p)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """Run forward pass.

        Parameters:
        - x (torch.Tensor): Input tensor of shape `(n_samples, n_patches + 1, dim)`.

        Returns:
        - torch.Tensor: Output tensor of shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        # Check if input dimension matches expected dimension
        if dim != self.dim:
            raise ValueError

        # Linear projections and reshaping for query, key, and value
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)

        # Attention mechanism calculations
        dp = q @ k_t * self.scale
        self.attn = dp.softmax(dim=-1)
        self.attn = self.attn_drop(self.attn)

        weighted_avg = self.attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)

        # Linear projection and dropout for the output
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc1 : nn.Linear
        The first linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(x)     # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)     # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)    # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)     # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)    # (n_samples, n_patches + 1, out_features)

        return x
