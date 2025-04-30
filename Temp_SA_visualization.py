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



def visualize_temp_map(model, img_tensor,save_path , class_names=None):
    """
    Visualizes the temperature map for a single image using a trained ViT with SelectiveAttention
    """
    
    model.eval()

    # CIFAR-10 class names if not provided
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(16, 10))

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        patch_embeddings = model.patch_embed(img_tensor)  # [1, num_patches, embed_dim]

        cls_token = model.cls_token.expand(patch_embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_token, patch_embeddings), dim=1)

        embeddings = embeddings + model.pos_embed
        embeddings = model.pos_drop(embeddings)

        temps_q = []
        temps_v = []

        for i, block in enumerate(model.blocks):
            # Apply layer normalization before attention
            norm_embeddings = block.norm1(embeddings)
            
            token_temp_q , token_temp_v = block.attn.token_aware_temperatures(norm_embeddings) #(1 , seq_len)
            pos_temp = block.attn.position_aware_temperatures(norm_embeddings) #(1,seq_len)

            q_temp = token_temp_q + pos_temp
            v_temp = token_temp_v + pos_temp

            temps_q.append(q_temp.cpu().numpy())
            temps_v.append(v_temp.cpu().numpy())

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

    patch_size = model.patch_embed.patch_size
    n_patches_1d = img_tensor.shape[-1] // patch_size

    # Select a subset of layers to visualize (first, middle, and last)
    selected_layers = [len(model.blocks) - 3 , len(model.blocks) -2 , len(model.blocks)-1]

    # Find global min and max for consistent colormap scaling
    q_min = min([temps[0 , 1:].min() for temps in temps_q])
    q_max = max([temps[0 , 1:].max() for temps in temps_q])
    normq = Normalize(vmin=q_min, vmax=q_max)

    v_min = min([temps[0 , 1:].min() for temps in temps_v])
    v_max = max([temps[0 , 1:].max() for temps in temps_v])
    normv = Normalize(vmin=v_min, vmax=v_max)

    # Visualize temperature maps for selected layers
    cmap_overlay = 'hot'

    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 2 + idx)

        temps = temps_v[layer_idx][0 , 1:]
        temps_map = temps.reshape(n_patches_1d, n_patches_1d)

        # Upsample attention map to match image size
        upsampled_temp = np.repeat(np.repeat(temps_map, patch_size, axis=0), patch_size, axis=1)

        # Create a figure with the image and attention overlay
        plt.imshow(img)
        im = plt.imshow(upsampled_temp, alpha=0.6, cmap=cmap_overlay, interpolation='lanczos', norm=normv)
        plt.title(f"Layer {layer_idx+1} Value Temp Overlay")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)


    for idx, layer_idx in enumerate(selected_layers):
        ax = plt.subplot(2, 5, 7 + idx)

        temps = temps_q[layer_idx][0 , 1:]
        temps_map = temps.reshape(n_patches_1d, n_patches_1d)

        # Upsample attention map to match image size
        upsampled_temp = np.repeat(np.repeat(temps_map, patch_size, axis=0), patch_size, axis=1)

        # Create a figure with the image and attention overlay
        plt.imshow(img)
        im = plt.imshow(upsampled_temp, alpha=0.6, cmap=cmap_overlay, interpolation='lanczos', norm=normq)
        plt.title(f"Layer {layer_idx+1} Query Temp Overlay")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)


    plt.suptitle(f"Predicted: {class_names[predicted_class]}", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.show()

def visualize_cifar10_example(experiment_model, index=None):
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

    print(f"True label: {class_names[label]}")
    print(f"Selected image index: {index}")

    visualize_temp_map(experiment_model, img_tensor, save_path = f"{index}_Temp_SA.png", class_names = class_names)
