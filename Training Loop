num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0

    for x0, y in train_loader:
        optimizer.zero_grad()
        x0 = x0.to(device)
        y  = y.to(device)

        # sample target from Gaussian noise
        x1 = torch.randn_like(x0)

        # sample random time t [0,1]
        t = torch.rand(x0.size(0), device=device)

        # Compute CFM flow (returns t_out, x_t, v_t)
        t_out, x_t, v_t = FM.sample_location_and_conditional_flow(x0, x1, t)

        # Model prediction
        x_t = x_t.view(x_t.size(0), -1)
        v_t = v_t.view(v_t.size(0), -1)
        t_out = t_out.view(x_t.size(0), 1)  # ensure (B,1)
        v_hat = model(t_out, x_t, y) # (B,784)
        # Loss & backward pass
        loss = F.mse_loss(v_hat, v_t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}  Loss: {total_loss:.4f}")
    # Generate 16 MNIST digits
    grid = generate_grid(model, FM, device, num_samples=16, steps=50)

    # Show the grid
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0).cpu())
    plt.axis('off')
    plt.show()
