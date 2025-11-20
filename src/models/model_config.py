from typing import Iterable, Dict, Set
import numpy as np
import torch
from models.model import *
from models.black_scholes import BlackScholesModel
from request_interface.request_interface import AtomicRequest

class ModelConfig(Model):
    """
    Model Configuration: Simulates several correlated models simultaneously.

    Args:
        models (list[Model]): Models to be simulated simultaneously. All must share the same calibration_date.
        correlation_matrix (list[np.ndarray],  None): list of (num_asset1 x num_asset2) inter asset correlations. 
        If None, identity is used.
    """

    def __init__(
        self,
        models: list[Model],
        numeraire_model_idx: int = 0,
        discount_model_idx: int = 0,
        inter_asset_correlation_matrix: list[np.ndarray] | None = None,
    ):
        assert len(models) > 0, "Provide at least one model."
        assert all(
            models[i].calibration_date == models[i + 1].calibration_date
            for i in range(len(models) - 1)
        ), "All models must share the same calibration_date."
        
        simulation_dim = sum(m.simulation_dim for m in models)
        # Ensure all asset_ids across models are unique
        asset_ids = [asset_id for model in models for asset_id in model.asset_ids]
        assert len(asset_ids) == len(set(asset_ids)), \
            "Duplicate asset_ids detected across models. A particular asset can only be simulated by one distinct model."
        super().__init__(
            calibration_date=models[0].calibration_date,
            asset_ids=asset_ids,
            simulation_dim=simulation_dim
            )

        self.models: list[Model] = models
        
        self.id_to_model: dict[str, int | None] = {
            "numeraire": numeraire_model_idx,
            "discount": discount_model_idx,
            }
        
        for idx, model in enumerate(models):
            for asset_id in model.asset_ids:
                self.id_to_model[asset_id] = idx
                
        self.model_state_offset: dict[int, int] = {}
        offset = 0
        for idx, model in enumerate(models):
            self.model_state_offset[idx] = offset
            offset += model.state_dim

        # collect shapes
        shapes = [len(model.get_model_params()) for model in models]
        total_dim = sum(shapes)
        self.model_params = torch.zeros(total_dim, dtype=FLOAT, device=device)
            
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
    
    def _get_correlation_matrix(self) -> torch.Tensor:
        """
        Joint correlation matrix, assembled block-wise:
        C = diag-blocks (intra) + off-diagonal blocks (inter).
        """
        num_assets = self.num_assets
        corr = torch.zeros((num_assets, num_assets), device=device, dtype=FLOAT)

        row = 0
        idx = 0
        for i, model1 in enumerate(self.models):
            num_assets1 = model1.num_assets

            # diagonal (intra) block
            corr[row:row+num_assets1, row:row+num_assets1] = model1._get_correlation_matrix()

            # off-diagonal (inter) blocks for j > i
            col = row + num_assets1
            for model2 in self.models[i+1:]:
                num_assets2 = model2.num_assets

                int_corr = self.inter_asset_correlation_matrix[idx]  
                
                # Write upper-right block
                corr[row:row+num_assets1, col:col+num_assets2] = int_corr

                # Write lower-left block (symmetric)
                if int_corr.ndim >= 2:
                    # transpose last two dims for any >=2D tensor
                    corr[col:col+num_assets2, row:row+num_assets1] = int_corr.transpose(-1, -2)
                else:
                    # 0-D or 1-D: transpose is a no-op, just reuse
                    corr[col:col+num_assets2, row:row+num_assets1] = int_corr  

                col += num_assets2

            row += num_assets1

        # numerical symmetrization (cheap safety against tiny asymmetries)
        corr = 0.5 * (corr + corr.T)
        return corr    

    def _get_covariance_matrix(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Joint covariance for one step Δt, assembled block-wise:
        C = diag-blocks (intra) + off-diagonal blocks (inter).
        """
        num_assets = self.num_assets
        cov = torch.zeros((num_assets, num_assets), device=device, dtype=FLOAT)

        row = 0
        idx = 0
        for i, model1 in enumerate(self.models):
            num_assets1 = model1.num_assets

            # diagonal (intra) block
            cov[row:row+num_assets1, row:row+num_assets1] = model1._get_covariance_matrix(delta_t)

            # off-diagonal (inter) blocks for j > i
            col = row + num_assets1
            for model2 in self.models[i+1:]:
                num_assets2 = model2.num_assets

                int_corr = self.inter_asset_correlation_matrix[idx]  
                block = self._compute_inter_covariance_matrix(
                    model1=model1,
                    model2=model2,
                    int_corr_mat=int_corr,
                    delta_t=delta_t,
                )  

                # Write upper-right block
                cov[row:row+num_assets1, col:col+num_assets2] = block

                # Write lower-left block (symmetric)
                if block.ndim >= 2:
                    # transpose last two dims for any >=2D tensor
                    cov[col:col+num_assets2, row:row+num_assets1] = block.transpose(-1, -2)
                else:
                    # 0-D or 1-D: transpose is a no-op, just reuse
                    cov[col:col+num_assets2, row:row+num_assets1] = block  

                col += num_assets2

            row += num_assets1

        # numerical symmetrization (cheap safety against tiny asymmetries)
        cov = 0.5 * (cov + cov.T)
        return cov

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

        raise NotImplementedError(
            "Inter covariance not implemented for the requested pair of models."
        )

    def simulate_time_step_analytically(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor                                       
    ) -> torch.Tensor:
        """
        state:  (N, markov_dim) or (markov_dim,)
        randn:  same shape as state (noise already correlated)
        """
        new_state = torch.zeros_like(state)
        start_state = 0
        start_sim = 0
        for model in self.models:
            end_state = start_state + model.state_dim
            end_sim = start_sim + model.simulation_dim
            new_state[:, start_state:end_state] = model.simulate_time_step_analytically(
                time1=time1,
                time2=time2,
                state=state[:, start_state:end_state], 
                corr_randn=corr_randn[:, start_sim:end_sim],
            )
            start_state = end_state
            start_sim = end_sim

        return new_state

    def simulate_time_step_euler(
        self,
        time1: torch.Tensor,
        time2: torch.Tensor, 
        state: torch.Tensor, 
        corr_randn: torch.Tensor     
    ) -> torch.Tensor:
        """
        Euler–Maruyama step. Calls the sub-models' *Euler* step.
        """
        new_state = torch.zeros_like(state)
        start_state = 0
        start_sim = 0
        for model in self.models:
            end_state = start_state + model.state_dim
            end_sim = start_sim + model.simulation_dim
            new_state[:, start_state:end_state] = model.simulate_time_step_euler(
                time1=time1,
                time2=time2,
                state=state[:, start_state:end_state], 
                corr_randn=corr_randn[:, start_sim:end_sim],
            )
            start_state = end_state
            start_sim = end_sim

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
        model_idx = self.id_to_model[asset_id]
        model = self.models[model_idx]
        start = self.model_state_offset[model_idx]

        if model.state_dim == 1:
            # Direct slice gives shape (num_paths,)
            model_state = state[:, start]
        else:
            # Multi-asset model: extract its block
            end = start + model.state_dim
            model_state = state[:, start:end]

        return model.resolve_request(req, asset_id, model_state)
