import torch
import torch.nn as nn

class EeqToZConditional(nn.Module):
    """
    Modifica un z_base (por nodo) condicionándolo al valor deseado de E_eq.
    """
    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_base, E_eq_scalar):
        """
        z_base: [B*N, latent_dim]  — vector latente base por nodo para cada muestra
        E_eq_scalar: [B, 1] — valores deseados del módulo elástico

        Devuelve: z_modificado: [B*N, latent_dim]
        """
        B = E_eq_scalar.size(0)
        N = z_base.size(0) // B

        e_expanded = E_eq_scalar.repeat_interleave(N, dim=0)  # [B*N, 1]
        input_concat = torch.cat([z_base, e_expanded], dim=1)  # [B*N, latent_dim + 1]
        return self.mlp(input_concat)  # [B*N, latent_dim]
