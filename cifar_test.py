from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("Training samples:", len(train_dataset))