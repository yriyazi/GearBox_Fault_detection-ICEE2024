import torch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
        Define a PatchEmbed class for splitting an image into patches and embedding them using convolutional layers.
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int=3, embedding_dimension: int=384) -> None:
        """
        Initialize the PatchEmbed module.

        Parameters:
        - img_size (int): Size of the input image (assumed to be square).
        - patch_size (int): Size of each square patch.
        - in_chans (int): Number of input channels.
        - embedding_dimension (int): The embedding dimension.

        Attributes:
        - n_patches (int): Number of patches inside the image.
        - proj (nn.Conv2d): Convolutional layer for both patch splitting and embedding.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Initialize the convolutional layer for patch splitting and embedding
        self.proj = nn.Conv2d(
            in_chans,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): Input tensor of shape `(Batch_size, in_chans, img_size, img_size)`.

        Returns:
        - torch.Tensor: Output tensor of shape `(Batch_size, n_patches, embedding_dimension)`.
        """
        x = self.proj(x)  # Apply the convolutional layer for patch splitting and embedding
        # Reshape the tensor to have shape `(Batch_size, n_patches, embedding_dimension)`
        x = x.flatten(2).transpose(1, 2)
        return x
