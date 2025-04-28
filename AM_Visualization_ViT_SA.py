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


def visualize_attention_map(model, img_tensor,save_path , class_names=None):
    """
    Visualizes the attention map for a single image using a trained ViT with SelectiveAttention
    """
    
    model.eval()

    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(16, 10))

    
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Get patch embeddings
    with torch.no_grad():
        # Forward pass through the patch embedding layer
        patch_embeddings = model.patch_embed(img_tensor)  # [1, num_patches, embed_dim]

        # Add class token
        cls_token = model.cls_token.expand(patch_embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_token, patch_embeddings), dim=1)

        # Add position embedding
        embeddings = embeddings + model.pos_embed
        embeddings = model.pos_drop(embeddings)

        # Store attention maps from each layer
        attention_maps = []

        # Collect attention maps from each transformer block
        for i, block in enumerate(model.blocks):
            # Apply layer normalization before attention
            norm_embeddings = block.norm1(embeddings)

            # Get attention map
            attn_map = block.attn.attention_map(norm_embeddings)  # [1, seq_len, seq_len]
            attention_maps.append(attn_map.cpu().numpy())

            # Forward through the block for next iteration
            embeddings = block(embeddings)

    # Display original image
    ax = plt.subplot(2, 5, 1)
    img = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    # Normalize image for display
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Get model prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # Calculate number of patches
    patch_size = model.patch_embed.patch_size
    n_patches_1d = img_tensor.shape[-1] // patch_size

    # Select a subset of layers to visualize (first, middle, and last)
    selected_layers = [len(model.blocks) - 3 , len(model.blocks) -2 , len(model.blocks)-1]

    # Find global min and max for consistent colormap scaling
    attn_min = min([maps[0, 0, 1:].min() for maps in attention_maps])
    attn_max = max([maps[0, 0, 1:].max() for maps in attention_maps])
    norm = Normalize(vmin=attn_min, vmax=attn_max)

    # Visualize attention maps for selected layers
    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 2 + idx)

        # Get attention map for CLS token (first token's attention to all other tokens)
        attn = attention_maps[layer_idx][0, 0, 1:]  # [num_patches]

        # Reshape to 2D grid for visualization
        attn_map = attn.reshape(n_patches_1d, n_patches_1d)

        # Upsample attention map to image size
        im = plt.imshow(attn_map, interpolation='lanczos', cmap='viridis', norm=norm)
        plt.title(f"Layer {layer_idx+1} - CLS Token Attention")
        plt.axis('off')

        # Add a colorbar
        plt.colorbar(im, fraction=0.046, pad=0.04)

    
    # Create heatmap overlays for the second row
    cmap_overlay = 'hot'

    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 7 + idx)

        # Get attention map and reshape
        attn = attention_maps[layer_idx][0, 0, 1:]
        attn_map = attn.reshape(n_patches_1d, n_patches_1d)

        # Upsample attention map to match image size
        upsampled_attn = np.repeat(np.repeat(attn_map, patch_size, axis=0), patch_size, axis=1)

        # Create a figure with the image and attention overlay
        plt.imshow(img)
        im = plt.imshow(upsampled_attn, alpha=0.6, cmap=cmap_overlay, interpolation='lanczos', norm=norm)
        plt.title(f"Layer {layer_idx+1} Attention Overlay")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)

   

    # Add overall title
    plt.suptitle(f"Predicted: {class_names[predicted_class]}", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.show()


def visualize_cifar10_example(experiment_model, index=None):

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

    print(f"True label: {class_names[label]}")
    print(f"Selected image index: {index}")

    # Visualize attention
    visualize_attention_map(experiment_model, img_tensor, save_path = f"{index}_AM_SA.png", class_names = class_names)


