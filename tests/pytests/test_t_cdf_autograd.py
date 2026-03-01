import torch
import numpy as np
from src.maths.pytorch_external_functions import t_cdf_autograd

def finite_diff_grad(f, x, eps=1e-5):
    """
    Central finite difference gradient for scalar f(x).
    eps=1e-5 is typically more stable when SciPy/CPU is involved.
    """
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_grad = grad.view(-1)

    for i in range(flat_x.numel()):
        x_pos = flat_x.clone()
        x_neg = flat_x.clone()
        x_pos[i] += eps
        x_neg[i] -= eps

        f_pos = f(x_pos.view_as(x)).item()
        f_neg = f(x_neg.view_as(x)).item()

        flat_grad[i] = (f_pos - f_neg) / (2 * eps)

    return grad

def build_correlation_from_unconstrained(theta, dim):
    """
    theta: unconstrained vector of length dim*(dim+1)//2
    returns: correlation matrix (dim x dim), guaranteed SPD
    """
    L = torch.zeros(dim, dim, dtype=theta.dtype, device=theta.device)

    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                # diagonal must be positive
                L[i, j] = torch.exp(theta[idx])
            else:
                L[i, j] = theta[idx]
            idx += 1

    # covariance
    Sigma = L @ L.T

    # normalize to correlation
    d = torch.sqrt(torch.diag(Sigma))
    Sigma_corr = Sigma / (d[:, None] * d[None, :])

    return Sigma_corr

def test_student_t_cdf_gradient_with_chol():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"
    dtype = torch.float64
    df = 3.7
    dim = 3

    # input: independent normals
    Z = torch.randn(7, dim, device=device, dtype=dtype, requires_grad=True)

    # correlation matrix (fixed, PD)
    rho = 0.4
    Sigma = torch.tensor(
        [[1.0, rho, 0.2],
         [rho, 1.0, 0.1],
         [0.2, 0.1, 1.0]],
        device=device,
        dtype=dtype,
    )
    chol = torch.linalg.cholesky(Sigma)

    def loss_fn(z):
        X = z @ chol.T
        U = t_cdf_autograd(X, df)
        return U.sum()

    # autograd gradient
    loss = loss_fn(Z)
    loss.backward()
    grad_autograd = Z.grad.detach().clone()

    # finite-difference gradient
    grad_fd = finite_diff_grad(loss_fn, Z.detach(), eps=1e-5)

    # comparison
    abs_err = (grad_autograd - grad_fd).abs()
    rel_err = abs_err / (grad_fd.abs() + 1e-12)

    print("Max abs error:", abs_err.max().item())
    print("Max rel error:", rel_err.max().item())

    # tolerances: with SciPy in the loop, 1e-5 is a sensible default
    assert abs_err.max() < 5e-5
    assert rel_err.max() < 5e-5
    
def test_student_t_cdf_gradient_wrt_corr_params_pd():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"
    dtype = torch.float64
    df = 3.7
    dim = 3

    # number of unconstrained parameters for Cholesky
    n_params = dim * (dim + 1) // 2
    theta = torch.randn(n_params, device=device, dtype=dtype, requires_grad=True)

    # fixed base samples
    Zbase = torch.randn(7, dim, device=device, dtype=dtype)

    def loss_fn(th):
        Sigma = build_correlation_from_unconstrained(th, dim)
        chol = torch.linalg.cholesky(Sigma)

        X = Zbase @ chol.T
        U = t_cdf_autograd(X, df)

        return U.sum()

    # autograd
    loss = loss_fn(theta)
    loss.backward()
    grad_autograd = theta.grad.detach().clone()

    # finite differences
    grad_fd = finite_diff_grad(loss_fn, theta.detach(), eps=1e-6)

    atol = 1e-8
    rtol = 1e-5  

    ok = torch.allclose(grad_autograd, grad_fd, atol=atol, rtol=rtol)

    print("allclose:", ok)
    assert ok