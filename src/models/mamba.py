import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaLite(nn.Module):
    """
    Prosta implementacja bloku inspirowanego Mambą, w czystym PyTorch.
    Wejście:  x o kształcie (B, L, D)
    Wyjście:  tensor o tym samym kształcie (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner = d_model * expand

        self.d_model = d_model
        self.d_state = d_state
        self.inner = inner

        self.norm = nn.LayerNorm(d_model)

        self.in_proj = nn.Linear(d_model, inner * 3, bias=True)

        self.conv = nn.Conv1d(
            in_channels=inner,
            out_channels=inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=inner,
            bias=True,
        )

        self.A_log = nn.Parameter(torch.randn(inner, d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(inner, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(inner, d_state) * 0.02)
        self.D = nn.Parameter(torch.zeros(inner))

        self.out_proj = nn.Linear(inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        residual = x

        x = self.norm(x)

        uvg = self.in_proj(x)  
        u, v, g = uvg.chunk(3, dim=-1)  

        u = self.conv(u.transpose(1, 2)).transpose(1, 2)  
        u = F.silu(u)

        # stabilna wersja A
        A = -torch.exp(self.A_log)      
        eA = torch.exp(A)               

        # stan ukryty
        s = torch.zeros(
            (B, self.inner, self.d_state),
            device=x.device,
            dtype=x.dtype,
        )

        ys = []
        for t in range(L):
            ut = u[:, t, :]  

            s = s * eA.unsqueeze(0) + ut.unsqueeze(-1) * self.B.unsqueeze(0)
            yt = (s * self.C.unsqueeze(0)).sum(dim=-1) + ut * self.D.unsqueeze(0)

            ys.append(yt)

        y = torch.stack(ys, dim=1)  

        # gating
        y = y * torch.sigmoid(g) + F.silu(v)

        y = self.out_proj(y)  
        y = self.dropout(y)

        return residual + y