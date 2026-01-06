def generate_grid(model, FM, device, num_samples=16, steps=50):
    # Start from Gaussian noise in image shape
    x = torch.randn(num_samples, 28*28, device=device)
    labels = torch.arange(num_samples) % 10
    y = labels.to(device)

    # Euler integration
    for step in range(steps):
        t = torch.full((num_samples,1), step/steps, device=device)
        v = model(t, x, y) # (B,784) output
        x = x + v / steps

    # reshape to (B,1,28,28) for visualization
    x_img = x.view(num_samples,28,28).unsqueeze(1) # (B,1,28,28)
    grid = make_grid(x_img, nrow=4, normalize=True)
    return grid

# Generate 16 MNIST digits
grid = generate_grid(model, FM, device, num_samples=16, steps=50)

# Show the grid
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0).cpu())
plt.axis('off')
plt.show()
