# This code is taken from the original S4 repository https://github.com/HazyResearch/state-spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


# --- replace the S4DKernel class with this fixed version ---
class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # use explicit complex64
        C = torch.randn(H, N // 2, dtype=torch.complex64)
        # store the real representation (float32) so view_as_complex works even if autocast changes dtype
        self.C = nn.Parameter(_c2r(C).to(torch.float32))
        self.register("log_dt", log_dt.to(torch.float32), lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2, dtype=torch.float32))
        # make A_imag float32
        A_imag = (math.pi * repeat(torch.arange(N // 2, dtype=torch.float32), 'n -> h n', h=H))
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Force computations to float32 to avoid bfloat16 -> view_as_complex error
        # Materialize parameters (as float32)
        dt = torch.exp(self.log_dt.to(torch.float32))  # (H)

        # ensure the real view is float32 before converting back to complex
        C_real = self.C
        if C_real.dtype != torch.float32:
            C_real = C_real.to(torch.float32)
        C = _r2c(C_real)  # complex64 (H, N)

        A_real = self.log_A_real.to(torch.float32)
        A_im = self.A_imag.to(torch.float32)
        A = -torch.exp(A_real) + 1j * A_im  # complex64 (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N) complex64
        # K: (H N L)
        # build time indices in float32 then exponentiate (complex)
        time_idx = torch.arange(L, device=A.device, dtype=torch.float32)
        K = dtA.unsqueeze(-1) * time_idx  # broadcasting -> complex
        # C multiply and sum
        # ensure C is complex64 and use complex-valued exp
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real  # (H, L), real32

        return K


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

def safe_rfft(k, u, L):
    torch.cuda.synchronize()
    torch.backends.cuda.cufft_plan_cache.clear()

    k = k.contiguous().float()
    u = u.contiguous().float()

    k_f = torch.fft.rfft(k, n=2 * L)
    u_f = torch.fft.rfft(u, n=2 * L)
    return k_f, u_f


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )



    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        # k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        # u_f = torch.fft.rfft(u.to(torch.float32), n=2 * L)  # (B H L)
        k_f, u_f = safe_rfft(k, u, L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Model(nn.Module):
    def __init__(self, in_dim, dropout, act):
        super(S4Model, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)
        self.s4_block = nn.Sequential(nn.LayerNorm(512),
                                      S4D(d_model=512, d_state=32, transposed=False))

    def forward(self, x, return_features=False):
        x = x.unsqueeze(0)

        x = self._fc1(x)
        x = self.s4_block(x)
        if return_features:
            return x, torch.max(x, axis=1).values
        else:
            return torch.max(x, axis=1).values
