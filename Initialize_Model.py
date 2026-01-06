# Initalization of Model
model = MLP(img_size=28, hidden=512, num_classes=10, class_cond=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler()
FM = ConditionalFlowMatcher(sigma=0.1)
print("Model, optimizer, and CFM matcher initialized.")
