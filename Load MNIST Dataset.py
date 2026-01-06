batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

print("MNIST loaded. Shape examples:")
x, y = next(iter(train_loader))
print("Image batch:", x.shape, "Labels:", y.shape)
