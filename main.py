import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import einops
import os

# Import the model components and tokenizer
from Selective_Attention_Module import SelectiveAttentionModule
from Encoder_Only_Block import EncoderOnlyBlock
from Tokenize_Dataset_Vision import enhanced_tokenize_image
from Architecture import EncoderOnlyArchitecture

# CIFAR-10 dataset preparation
def prepare_cifar10_datasets(patch_size=4, num_tokens=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # Tokenize the datasets
    train_tokenized = []
    for image, label in train_dataset:
        tokens = enhanced_tokenize_image(image, patch_size=patch_size, num_tokens=num_tokens)
        train_tokenized.append((tokens, label))
    
    test_tokenized = []
    for image, label in test_dataset:
        tokens = enhanced_tokenize_image(image, patch_size=patch_size, num_tokens=num_tokens)
        test_tokenized.append((tokens, label))
    
    return train_tokenized, test_tokenized

# Custom collate function for batching
def collate_fn(batch):
    tokens, labels = zip(*batch)
    return torch.stack([t for t in tokens]), torch.tensor(labels)

# Training function
def train(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for tokens, labels in progress_bar:
            tokens, labels = tokens.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Adjust learning rate
        scheduler.step(test_loss)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_results.png')
    plt.close()
    
    return train_losses, test_losses, test_accuracies

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for tokens, labels in data_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Main training script
def main():
    # Hyperparameters
    patch_size = 4  # 4x4 patches
    num_tokens = 512  # Vocabulary size
    embedding_dim = 256  # Embedding dimension
    num_classes = 10  # CIFAR-10 has 10 classes
    num_encoder_blocks = 6  # Number of encoder blocks
    batch_size = 64
    epochs = 15
    lr = 0.001
    
    # Calculate sequence length (number of patches per image)
    seq_len = (32 // patch_size) ** 2  # CIFAR images are 32x32
    
    # Prepare datasets
    print("Preparing datasets...")
    train_data, test_data = prepare_cifar10_datasets(patch_size, num_tokens)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = EncoderOnlyArchitecture(
        n_cats=num_classes,
        vocab_size=num_tokens,
        embd_dim=embedding_dim,
        seq_len=seq_len,
        num_encoder_blocks=num_encoder_blocks
    ).to(device)
    
    # Print model summary
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    train_losses, test_losses, test_accuracies = train(
        model, train_loader, test_loader, epochs=epochs, lr=lr, device=device
    )
    
    # Save the model
    torch.save(model.state_dict(), 'cifar10_encoder_model.pth')
    
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")

if __name__ == "__main__":
    main()