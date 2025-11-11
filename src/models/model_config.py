from typing import Iterable, Dict, Set
import numpy as np
import torch
from models.model import *
from models.black_scholes import BlackScholesModel
from models.vasicek import VasicekModel
from models.cirpp import CIRppModel
from request_interface.request_interface import AtomicRequest, AtomicRequestType

class ModelConfig(Model):
    """
    Model Configuration: Simulates several correlated models simultaneously.

    Args:
        models (list[Model]): Models to be simulated simultaneously. All must share the same calibration_date.
        correlation_matrix (list[np.ndarray],  None): list of (num_asset1 x num_asset2) inter asset correlations. If None, identity is used.
    """

    def __init__(
        self,
        models: list[Model],
        inter_asset_correlation_matrix: list[np.ndarray] | None,
    ):
        assert len(models) > 0, "Provide at least one model."
        assert all(
            models[i].calibration_date == models[i + 1].calibration_date
            for i in range(len(models) - 1)
        ), "All models must share the same calibration_date."

        super().__init__(calibration_date=models[0].calibration_date,asset_ids=[])

        self.models = models
        self.markov_dim = sum(m.markov_dim for m in models)
        self.num_assets = sum(m.num_assets for m in models)
        
        self.asset_to_model: dict[str, int] = {"numeraire": 0}
        for idx, model in enumerate(models):
            for asset_id in model.asset_ids:
                self.asset_to_model[asset_id] = idx
                
        self.model_state_offset: dict[int, int] = {}
        offset = 0
        for idx, model in enumerate(models):
            self.model_state_offset[idx] = offset
            offset += model.num_assets

        # collect shapes
        shapes = [len(model.get_model_params()) for model in models]
        total_dim = sum(shapes)
        self.model_params = torch.zeros(total_dim, dtype=FLOAT, device=device)

        # link each model's params as a view
        offset = 0
        self._param_views = []
        for model, n in zip(models, shapes):
            view = self.model_params[offset:offset+n]          # <-- view, no copy
            model.set_model_params(view)                  # model must store reference
            self._param_views.append(view)
            offset += n
            
        self.inter_asset_correlation_matrix: list[torch.Tensor] = []

        if inter_asset_correlation_matrix is None:
            for i, model1 in enumerate(models):
                for model2 in models[i + 1:]:
                    self.inter_asset_correlation_matrix.append(
                        torch.zeros(model1.num_assets, model2.num_assets, dtype=FLOAT, device=device)
                    )
        else:
            for corr_mat in inter_asset_correlation_matrix:
                self.inter_asset_correlation_matrix.append(
                    torch.tensor(corr_mat, dtype=FLOAT, device=device)
                )

        
        self.cholesky: dict[float, torch.Tensor] = {}
        
    def get_state(self, num_paths: int) -> torch.Tensor:
        """
        Collect log-states from all submodels and concatenate into one big tensor.
        Returns shape (num_paths, total_markov_dim)
        """
        states = []
        for model in self.models:
            s = model.get_state(num_paths)   
            if s.ndim == 1:
                s = s.unsqueeze(1)           
            states.append(s)
        return torch.cat(states, dim=1)      

    def compute_cov_matrix(self, delta_t: float | torch.Tensor) -> torch.Tensor:
        """
        Joint covariance for one step Δt, assembled block-wise:
        C = diag-blocks (intra) + off-diagonal blocks (inter).
        """
        dt = float(delta_t)
        num_assets = self.num_assets
        C = torch.zeros((num_assets, num_assets), device=device, dtype=FLOAT)

        row = 0
        idx = 0
        for i, model1 in enumerate(self.models):
            num_assets1 = model1.num_assets

            # diagonal (intra) block
            C[row:row+num_assets1, row:row+num_assets1] = model1.compute_cov_matrix(dt)

            # off-diagonal (inter) blocks for j > i
            col = row + num_assets1
            for j, model2 in enumerate(self.models[i+1:], start=i+1):
                num_assets2 = model2.num_assets

                int_corr = self.inter_asset_correlation_matrix[idx]  
                block = self._compute_inter_covariance_matrix(
                    model1=model1,
                    model2=model2,
                    int_corr_matrix=int_corr,
                    delta_t=dt,
                )  

                C[row:row+num_assets1, col:col+num_assets2] = block
                C[col:col+num_assets2, row:row+num_assets1] = block.T  

                col += num_assets2

            row += num_assets1

        # numerical symmetrization (cheap safety against tiny asymmetries)
        C = 0.5 * (C + C.T)
        return C

    def _get_cov_mat_BSBS(self, bs1: BlackScholesModel, bs2: BlackScholesModel, int_corr_mat: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
            # Expect shape: vol1 (n1,), vol2 (n2,), int_corr (n1, n2)
            vol1 = bs1.get_volatility()
            vol2 = bs2.get_volatility()

            # Efficient outer-product form avoids building diag matrices
            # block[a,b] = vol1[a] * vol2[b] * int_corr[a,b] * dt
            return torch.outer(vol1, vol2) * int_corr_mat * delta_t

    def _compute_inter_covariance_matrix(
        self,
        model1: Model,
        model2: Model,
        int_corr_mat: torch.Tensor,
        delta_t: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Inter-asset covariance between model1 and model2 for BS-style dynamics:
        Cov = (σ1 ⊗ σ2) ∘ ρ * Δt
        where (σ1 ⊗ σ2)[a,b] = σ1[a] * σ2[b], and ∘ is elementwise product.
        """
        dt = float(delta_t)

        if isinstance(model1, BlackScholesModel) and isinstance(model2, BlackScholesModel):
            return self._get_cov_mat_BSBS(bs1=model1,bs2=model2,int_corr_mat=int_corr_mat, delta_t=dt)
        
        if isinstance(model1, BlackScholesModel) and isinstance(model2, CIRppModel):
            pass

        raise NotImplementedError(
            "Inter covariance not implemented for the requested pair of models."
        )

    def compute_cholesky(self, delta_t: float | torch.Tensor) -> torch.Tensor:
        """
        Cache Cholesky by (float) Δt. Assumes positive definite covariance.
        """
        dt = float(delta_t)
        if dt not in self.cholesky:
            cov = self.compute_cov_matrix(dt)
            self.cholesky[dt] = torch.linalg.cholesky(cov)
        return self.cholesky[dt]
    
    def generate_correlated_randn(self, num_paths: int, delta_t: float) -> torch.Tensor:
        cholesky = self.compute_cholesky(delta_t=delta_t)
        z = torch.randn(num_paths, self.num_assets, dtype=FLOAT, device=device)
        return z @ cholesky.T

    def simulate_time_step_analytically(self, delta_t: float, state: torch.Tensor, corr_randn: torch.Tensor) -> torch.Tensor:
        """
        state:  (N, markov_dim) or (markov_dim,)
        randn:  same shape as state (noise already correlated)
        """
        new_state = torch.zeros_like(state)
        start = 0
        for model in self.models:
            end = start + model.markov_dim
            new_state[:, start:end] = model.simulate_time_step_analytically(
                delta_t, state[:, start:end], corr_randn[:, start:end]
            )
            start = end

        return new_state

    def simulate_time_step_euler(self, state: torch.Tensor, randn: torch.Tensor) -> torch.Tensor:
        """
        Euler–Maruyama step. Calls the sub-models' *Euler* step.
        """
        new_state = torch.zeros_like(state)
        start = 0
        for model in self.models:
            end = start + model.markov_dim
            new_state[:, start:end] = model.simulate_time_step_euler(
                state[:, start:end], randn[:, start:end]
            )
            start = end

        return new_state

    def _prepare_state(self, num_paths: int) -> torch.Tensor:
        """
        Returns an initial state of shape (num_paths, markov_dim).
        """
        spot = self.get_spot().reshape(1, -1)
        return spot.expand(num_paths, spot.size(1))

    def resolve_request(
        self, req: AtomicRequest, asset_id: str, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Resolve a request for a given asset from the concatenated global state.

        state: (num_paths, total_markov_dim)
        Returns: 1D tensor (num_paths,) for single-asset models,
                or (num_paths, model.num_assets) for multi-asset models.
        """
        model_idx = self.asset_to_model[asset_id]
        model = self.models[model_idx]
        start = self.model_state_offset[model_idx]

        if model.num_assets == 1:
            # Direct slice gives shape (num_paths,)
            model_state = state[:, start]
        else:
            # Multi-asset model: extract its block
            end = start + model.num_assets
            model_state = state[:, start:end]

        return model.resolve_request(req, asset_id, model_state)
