from torchvision import datasets, transforms

# 数据转换
transform = transforms.ToTensor()

# 下载 Fashion-MNIST
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 输出数据集大小
print("Training samples:", len(train_dataset))