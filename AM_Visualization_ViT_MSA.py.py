import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid
from matplotlib.colors import Normalize


def visualize_attention_maps(model, img_tensor, class_names=None, save_path="attention_maps.png"):
    """
    Visualizes attention maps for a single image using a ViT with Modified Selective Attention

    Args:
        model: Trained ViTWithModifiedSelectiveAttention model
        img_tensor: Single image tensor of shape [1, 3, H, W]
        class_names: List of class names (e.g., CIFAR-10 classes)
        save_path: Path to save the visualization
    """
    model.eval()

    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(16, 8))


    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)


    patch_size = model.patch_embed.patch_size
    n_patches_1d = img_tensor.shape[-1] // patch_size


    attention_maps = []

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()


        patch_embeddings = model.patch_embed(img_tensor)

        cls_token = model.cls_token.expand(patch_embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_token, patch_embeddings), dim=1)

        embeddings = embeddings + model.pos_embed

        if hasattr(model, 'pos_drop'):
            embeddings = model.pos_drop(embeddings)

        for block in model.blocks:
            norm_embeddings = block.norm1(embeddings)

            attn_map = block.attn.attention_map(norm_embeddings)
            attention_maps.append(attn_map.cpu().numpy())

            embeddings = block(embeddings)

    # Display original image
    ax = plt.subplot(2, 5, 1)
    img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Select layers to visualize 
    num_layers = len(model.blocks)
    selected_layers = [num_layers-3, num_layers - 2, num_layers - 1]

    attn_min = min([maps[0, 0, 1:].min() for maps in attention_maps])
    attn_max = max([maps[0, 0, 1:].max() for maps in attention_maps])
    norm = Normalize(vmin=attn_min, vmax=attn_max)

    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 2 + idx)

        attn = attention_maps[layer_idx][0, 0, 1:]  # [num_patches]

        attn_map = attn.reshape(n_patches_1d, n_patches_1d)

        im = plt.imshow(attn_map, interpolation='lanczos', cmap='viridis', norm=norm)
        plt.title(f"Layer {layer_idx+1} - CLS Token Attention")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)

    # Create attention overlays on original image
    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 7 + idx)

        attn = attention_maps[layer_idx][0, 0, 1:]
        attn_map = attn.reshape(n_patches_1d, n_patches_1d)

        upsampled_attn = np.repeat(np.repeat(attn_map, patch_size, axis=0), patch_size, axis=1)

        plt.imshow(img)
        im = plt.imshow(upsampled_attn, alpha=0.6, cmap='hot', interpolation='lanczos', norm=norm)
        plt.title(f"Layer {layer_idx+1} Attention Overlay")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)

    # Add overall title
    plt.suptitle(f"Predicted: {class_names[predicted_class]}", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.show()



def visualize_cifar10_example(model, index=None, save_path="attention_example.png"):
    """
    Visualizes the attention maps for a CIFAR-10 image

    Args:
        model: Trained ViTWithModifiedSelectiveAttention model
        index: Index of CIFAR-10 test image to visualize (random if None)
        save_path: Path to save the visualization
    """
    # Load CIFAR-10 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Select a random index if none provided
    if index is None:
        index = np.random.randint(0, len(testset))

    # Get image and label
    img_tensor, label = testset[index]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Visualize attention
    visualize_attention_maps(model, img_tensor, class_names, save_path)
