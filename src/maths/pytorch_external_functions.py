import torch
import scipy.special
import scipy.stats

class TCDFPytorch(torch.autograd.Function):
    """
    Forward: SciPy Student-t CDF on CPU (NumPy)
    Backward: exact derivative wrt X using Student-t PDF (SciPy), mapped back to torch
    """
    @staticmethod
    def forward(ctx, X: torch.Tensor, df: float):
        df = float(df)
        ctx.df = df
        ctx.save_for_backward(X)

        X_cpu = X.detach().cpu().numpy()
        U_cpu = scipy.special.stdtr(df, X_cpu)  # CDF
        U = torch.from_numpy(U_cpu).to(device=X.device, dtype=X.dtype)
        return U

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (X,) = ctx.saved_tensors
        df = ctx.df

        X_cpu = X.detach().cpu().numpy()
        pdf_cpu = scipy.stats.t.pdf(X_cpu, df)  # d/dx CDF = PDF
        pdf = torch.from_numpy(pdf_cpu).to(device=X.device, dtype=X.dtype)

        grad_X = grad_out * pdf
        return grad_X, None  


def t_cdf_autograd(X: torch.Tensor, df: float) -> torch.Tensor:
    return TCDFPytorch.apply(X, df)