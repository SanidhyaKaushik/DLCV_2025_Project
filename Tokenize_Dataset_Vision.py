import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import einops

transform = transforms.Compose([
    transforms.ToTensor(),  
])

dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def tokenize_image(image, patch_size=4):
    """
    Converts a CIFAR-10 image (3x32x32) into a sequence of patch indices.
    
    Args:
        image (torch.Tensor): A tensor of shape (3, 32, 32).
        patch_size (int): Size of the patch (e.g., 4 means 4x4 patches).
        
    Returns:
        patch_indices (torch.Tensor): A tensor of shape (num_patches,).
    """
    
    patches = einops.rearrange(image, "c (h p1) (w p2) -> (h w) c p1 p2", p1=patch_size, p2=patch_size)
    
    
    num_patches = patches.shape[0]
    patch_indices = torch.arange(num_patches)  
    
    return patch_indices