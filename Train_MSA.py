import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Compose
from tqdm import tqdm

# Training routine
def train_vit_cifar10(model, device, epochs=100, batch_size=128, lr=0.001, weight_decay=0.05):
    # Data augmentation and normalization
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616))
    ])

    transform_test = Compose([
        ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616))
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (inputs, targets) in enumerate(train_loop):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_loop.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
            for batch_idx, (inputs, targets) in enumerate(test_loop):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                test_loop.set_postfix({
                    'loss': test_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            }, f'vit_selective_attention_cifar10_epoch_{epoch+1}.pth')

    return history