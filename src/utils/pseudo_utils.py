import torch


# -----------------------------
# Utility functions
# -----------------------------
def procrustes_alignment(G, F, k=None):
    """
    Compute the Procrustes alignment between matrices A and B.
    G = (n,f) -> f == 768
    F = (n,d)
    """
    # 1) full SVD of the cross‐covariance
    M = F.T @ G  # (d,n)@(n,f)         # M: dxf
    U, _, V = torch.svd(M)  # U: d×d;   V: f×d

    # 2) optionally truncate to top‐k
    if k is not None:
        U_k = U[:, :k]  # d×k
        V_k = V[:, :k]  # f×k
    else:
        U_k = U  # d×d
        V_k = V  # f×d

    # 3) Procrustes alignment
    R_k = U_k @ V_k.T  # d×f

    return R_k.T, U_k, V_k


def linear_alignment(G, F, k=1):
    """
    Compute the linear alignment between matrices A and B.
    G = (n,f) -> f == 768
    F = (n,d)

    returns: R_k.T, U_k, V_k
    """
    # breakpoint()
    # 1. Compute the best linear mapping using least squares
    R = torch.linalg.lstsq(F, G).solution  # (f,d)
    # 2. SVD of the mapping for k components and using S**1/2 for the scaling
    U, S, Vt = torch.linalg.svd(R, full_matrices=False)
    S[k:] = 0
    S = torch.sqrt(torch.diag(S))
    U_k = (U @ S)[:, :k]
    V_k = (S @ Vt)[:k, :]
    return R, U_k, V_k.T


def neural_alignment(G, F, k):
    """
    Train a simple MLP: Linear -> Sigmoid -> Linear
    Ritorna: model=None (per compatibilità), E, D

    returns E, D
    """
    import torch
    from torch import nn, optim

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.Sigmoid()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x

    # --- setup ---
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = F.to(device)
    G = G.to(device)

    input_dim = F.size(1)
    output_dim = G.size(1)
    hidden_dim = int(k)

    # dataset
    n = F.size(0)
    dataset = torch.utils.data.TensorDataset(F, G)

    # ------- policy in base a n -------
    # soglie semplici e conservative
    VERY_SMALL_N = 20
    SMALL_N = 64

    if n <= VERY_SMALL_N:
        # Pochissimi sample: niente valida, full-batch, L2 aggressivo, poche epoche ma curate
        batch_size = n
        max_epochs = 300
        weight_decay = 1e-2
        lr = 1e-3
        use_val = False
    elif n <= SMALL_N:
        # Qualche decina di sample: mini holdout 80/20, early stopping
        batch_size = min(32, n)
        max_epochs = 600
        weight_decay = 3e-3  # un filo meno aggressivo
        lr = 1e-3
        use_val = True
        val_ratio = 0.2
    else:
        # 64+ sample: valida 10%, early stopping, un po' più di training
        batch_size = 64
        max_epochs = 800
        weight_decay = 1e-3
        lr = 1e-3
        use_val = True
        val_ratio = 0.1

    # split (solo se serve)
    if use_val:
        idx = torch.randperm(n, device=device)
        s = max(1, int(n * (1 - val_ratio)))
        tr_idx, va_idx = idx[:s], idx[s:]
        F_tr, G_tr = F[tr_idx], G[tr_idx]
        F_va, G_va = F[va_idx], G[va_idx]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(F_tr, G_tr),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        # valida full-batch
        val_pair = (F_va, G_va)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_pair = None

    # --- modello+opt ---
    model = SimpleMLP(input_dim, output_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler prudente: riduce LR se valida non migliora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=30,
        cooldown=0,
        verbose=False,
        min_lr=1e-5,
    )

    # early stopping solo se abbiamo valida
    best_state = None
    best_val = float("inf")
    patience = 80 if use_val else None
    no_improve = 0

    # --- training loop ---
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            # stabilità con pochi dati
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        # validazione
        if use_val:
            model.eval()
            with torch.no_grad():
                xva, yva = val_pair
                va_pred = model(xva)
                val_loss = criterion(va_pred, yva).item()
            scheduler.step(val_loss)

            if val_loss + 1e-10 < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            # senza valida: usa un decadimento del LR “a tempo”
            if (epoch + 1) % 100 == 0:
                for g in optimizer.param_groups:
                    g["lr"] = max(1e-5, g["lr"] * 0.5)

    # carica il best se abbiamo valida
    if use_val and best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # --- estrai E, D (come nel tuo codice originale) ---
    with torch.no_grad():
        E = model.fc1.weight.data.T.clone()  # (input_dim, hidden_dim)
        D = model.fc2.weight.data.clone()  # (hidden_dim, output_dim)

    # cleanup esplicito
    del model

    return None, E, D


def uniform_quantize(x: torch.Tensor, q: int) -> torch.Tensor:
    """
    Uniformly quantizes tensor x to q-bit precision and then dequantizes it.
    """
    x_min = -1
    x_max = 1
    if x_min == x_max:
        return x
    scale = (x_max - x_min) / (2**q - 1)
    x_int = torch.round((x - x_min) / scale)
    x_int = torch.clamp(x_int, 0, 2**q - 1)
    return x_int * scale + x_min


def white_noise(z, snr_db):
    """
    Add white noise to the input signal given an SNR in dB.
    """
    snr_linear = 10 ** (snr_db / 10)
    x_power = z.pow(2).mean()
    var = x_power / snr_linear
    std = torch.sqrt(var)
    return torch.randn_like(z) * std


def pseudo_inverse(anchs, relrep, regularized_inv):
    """
    Compute x_hat = relrep @ anchs @ regularized_inv.
    """
    A = anchs @ regularized_inv
    return relrep @ A


def frame_recon(anchs, relrep):
    """
    Simple multiplication: x_hat = relrep @ anchs.
    """
    return relrep @ anchs
