class MLP(nn.Module):
    def __init__(self, img_size=28, hidden=512, num_classes=10, class_cond=True, time_emb_dim=64):
        super().__init__()
        self.img_size = img_size

        self.flat_dim = img_size * img_size
        
        self.class_cond = class_cond

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        if class_cond:
            self.class_emb = nn.Embedding(num_classes, 32)
            cond_dim = 32
        else:
            cond_dim = 0

        # ----- Main MLP -----
        total_in = self.flat_dim + time_emb_dim + cond_dim

        layers = []
        layers.append(nn.Linear(total_in, hidden))
        layers.append(nn.SiLU())
        layers.append(nn.LayerNorm(hidden))

        for _ in range(5):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden))

        layers.append(nn.Linear(hidden, self.flat_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x, y=None):
        '''
        Args:
          t: (bs,1)
          x: (bs,flat_img)
          y: (bs,1)
        Output:

        '''
        batch = x.size(0)

        t_emb = self.time_mlp(t)

        # class conditioning
        if self.class_cond:
            if y is None:
                y = torch.zeros(batch, dtype=torch.long, device=x.device)
            y_emb = self.class_emb(y)
            cond = torch.cat([x,t_emb, y_emb], dim=1)
        else:
            cond = torch.cat([x,t_emb], dim=1)
        # process each row separately
        v = self.mlp(cond) 
        
        return v
