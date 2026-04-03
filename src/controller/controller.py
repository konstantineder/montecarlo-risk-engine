from common.packages import *
from common.enums import SimulationScheme
import logging
import time
from typing import List, Sequence
from engine.engine import MonteCarloEngine
from request_interface.request_interface import RequestInterface
from request_interface.request_types import AtomicRequest, AtomicRequestType
from metrics.metric import MetricType, Metric
from metrics.risk_metrics import PathwisePrimitive, RiskMetrics
from models.model import Model
from models.model_config import ModelConfig
from controller.simulation_results import SimulationResults
from products.netting_set import NettingSet
from products.product import Product
from collections import defaultdict
from maths.regression import RegressionFunction, PolyomialRegression

logger = logging.getLogger(__name__)

class SimulationController:
    """
    Simulation Controller to perform Monte Carlo simulation
    and compute metric outputs for each netting set.
    """
    def __init__(
        self, 
        netting_sets        : Sequence[NettingSet],    # Netting sets containing all products to be evaluated
        model               : Model,                  # Model used to simulate paths
        risk_metrics        : RiskMetrics,            # Collection of metrics used to compute outputs
        num_paths_mainsim   : int,                    # Number of Monte Carlo paths used for main simulation
        num_paths_presim    : int,                    # Number of Monte Carlo paths used for pre simulation
        num_steps           : int,                    # Number of time steps used for path simulation
        simulation_scheme   : SimulationScheme,       # Set Simulation Scheme: Schemes currently provided: Analytical, Euler, Milstein
        differentiate       : bool = False,           # Turn differentiation on or off
        regression_function : RegressionFunction = PolyomialRegression(degree=2),
    ):  
        self.risk_metrics = risk_metrics
        netting_sets = list(netting_sets)
        if len(netting_sets) == 0:
            raise ValueError("Provide at least one netting set.")

        seen_products: set[int] = set()
        for netting_set in netting_sets:
            for product in netting_set.products:
                product_id = id(product)
                if product_id in seen_products:
                    raise ValueError("A product instance cannot belong to more than one netting set.")
                seen_products.add(product_id)

        products = [product for netting_set in netting_sets for product in netting_set.products]
        self.netting_sets = netting_sets
        self.product_to_netting_set_idx: list[int] = []
        for netting_set_idx, netting_set in enumerate(self.netting_sets):
            self.product_to_netting_set_idx.extend([netting_set_idx] * len(netting_set.products))

        self.metric_exposure_timeline = self.risk_metrics.exposure_timeline.clone()
        self.exposure_timeline = self._build_internal_exposure_timeline()
        self._exposure_time_to_idx = {
            float(t.item()): idx for idx, t in enumerate(self.exposure_timeline)
        }
        if len(self.metric_exposure_timeline) > 0:
            self.metric_exposure_indices = torch.tensor(
                [self._exposure_time_to_idx[float(t.item())] for t in self.metric_exposure_timeline],
                dtype=torch.long,
                device=device,
            )
        else:
            self.metric_exposure_indices = torch.zeros(0, dtype=torch.long, device=device)
        self.netting_set_delayed_exposure_indices = self._build_netting_set_delayed_exposure_indices()
        
        # Set up requests for each exposure timepoint
        self.numeraire_requests: dict[tuple[float, str], AtomicRequest] = {
            (float(t.item()), "numeraire"): AtomicRequest(AtomicRequestType.NUMERAIRE, time1=t.item())
            for t in self.exposure_timeline
        }

        self.spot_requests: dict[tuple[float, str], AtomicRequest] = {
            (float(t.item()), asset_id): AtomicRequest(AtomicRequestType.SPOT)
            for prod in products
            for asset_id in prod.asset_ids
            for t in self.exposure_timeline
        }

        # If CVA is requested, ensure ModelConfig has a credit model and add requests to compute 
        # intermediate probabilities in terms of numeraire requests
        self.survival_prob_requests: dict[tuple[int, str], AtomicRequest] = {}
        self.cond_survival_prob_requests: dict[tuple[int, str], AtomicRequest] = {}
        if risk_metrics.any_xva:
            if not isinstance(model, ModelConfig):
                raise Exception("ModelConfig needs to be provided for xVA valuation.")

            counterparty_ids = risk_metrics.counterparty_ids
            if not all(
                counterparty in model.id_to_model for counterparty in counterparty_ids
            ):
                raise Exception("Not all models set for xVA valuation.")

        self.products = products
        self.model = model
        self.num_paths_presim = num_paths_presim
        self.num_paths_mainsim = num_paths_mainsim
        self.num_steps = num_steps
        self.simulation_scheme = simulation_scheme
        self.differentiate = differentiate
        self.regression_function=regression_function
        self.requires_higher_order_derivatives=False

        # Label products
        # Used to store and retrieve regression coefficients
        prod_id=0
        for prod in products:
            prod.product_id=prod_id
            prod_id+=1

        # Set differentiation mode
        # If enabled, differentiation of metric outputs is performed via AAD
        if differentiate:
            self.model.requires_grad()

        # Pre-allocate array to store regression coefficients
        # for each product and exposure timepoint
        self.regression_coeffs = []

        degree = self.regression_function.get_degree()
        num_time_points = len(self.exposure_timeline)

        for prod in products:
            prod._allocate_regression_coeffs(self.regression_function)
            num_states = prod.get_num_states()

            coeffs_tensor = torch.zeros(
                (num_time_points, num_states, degree),
                dtype=FLOAT,          # <- match the rest of your code
                device=device,
            )
            self.regression_coeffs.append(coeffs_tensor)

        # Collect timelines of each product and unify with exposure timeline
        # Remove duplicates and sort timeline
        # Store as common simulation timellne
        prod_times = {float(t.item()) for prod in self.products for t in prod.modeling_timeline}
        exposure_times = {t.item() for t in self.exposure_timeline}
        all_times = sorted(prod_times.union(exposure_times))
        self.simulation_timeline = torch.tensor(all_times, dtype=FLOAT, device=device)

        # Decide whether regression is required
        self.requires_regression = any(
            self._product_requires_regression(prod)
            for prod in self.products
        )

    def _build_internal_exposure_timeline(self) -> torch.Tensor:
        if not self.risk_metrics.requires_exposure_profiles():
            return self.risk_metrics.exposure_timeline.clone()

        exposure_times = {float(t.item()) for t in self.risk_metrics.exposure_timeline}
        for netting_set in self.netting_sets:
            if netting_set.is_collateralized():
                delayed_times = netting_set.get_collateral_query_times(self.risk_metrics.exposure_timeline)
                exposure_times.update(float(t.item()) for t in delayed_times)
        return torch.tensor(sorted(exposure_times), dtype=FLOAT, device=device)

    def _build_netting_set_delayed_exposure_indices(self) -> list[torch.Tensor]:
        delayed_indices_per_netting_set: list[torch.Tensor] = []
        num_metric_dates = len(self.metric_exposure_timeline)
        for netting_set in self.netting_sets:
            delayed_indices = torch.full(
                (num_metric_dates,),
                -1,
                dtype=torch.long,
                device=device,
            )
            if netting_set.is_collateralized():
                delayed_times = self.metric_exposure_timeline - netting_set.margin_period_of_risk
                valid = delayed_times >= 0.0
                if torch.any(valid):
                    delayed_indices[valid] = torch.tensor(
                        [
                            self._exposure_time_to_idx[float(t.item())]
                            for t in delayed_times[valid]
                        ],
                        dtype=torch.long,
                        device=device,
                    )
            delayed_indices_per_netting_set.append(delayed_indices)
        return delayed_indices_per_netting_set

    @staticmethod
    def _make_unique_names(base_names: list[str]) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        unique_names: list[str] = []

        for base_name in base_names:
            counts[base_name] += 1
            occurrence = counts[base_name]
            if occurrence == 1:
                unique_names.append(base_name)
            else:
                unique_names.append(f"{base_name}#{occurrence}")

        return unique_names

    def _product_requires_regression(self, product: Product) -> bool:
        if len(product.regression_timeline) > 0:
            return True
        if not self.risk_metrics.requires_exposure_profiles():
            return False
        return not self._can_use_analytic_exposure_for_product(product)

    def _can_use_analytic_exposure_for_product(self, product: Product) -> bool:
        supported_metric_types = {MetricType.PV, MetricType.EPE, MetricType.PFE}
        return (
            all(metric.metric_type in supported_metric_types for metric in self.risk_metrics.metrics)
            and product.supports_analytic_exposure(self.model)
        )

    def _can_evaluate_metric_analytically_for_product(
        self,
        product: Product,
        metric: Metric,
    ) -> bool:
        return (
            metric.metric_type == MetricType.PV
            and metric.evaluation_type == Metric.EvaluationType.ANALYTICAL
            and product.supports_analytic_pv(self.model)
        )

    def _can_skip_monte_carlo_for_product(self, product: Product) -> bool:
        if self.risk_metrics.requires_exposure_profiles():
            return False
        return all(
            self._can_evaluate_metric_analytically_for_product(product, metric)
            for metric in self.risk_metrics.metrics
        )

    def _get_requests(self) -> dict[tuple[int, str], set[AtomicRequest]]:
        """Collect and return all requests for exposure computation."""
        requests: dict[tuple[int, str], set[AtomicRequest]] = defaultdict(set)
        for label, req in self.numeraire_requests.items():
            requests[label].add(req)
            
        for label, req in self.spot_requests.items():
            requests[label].add(req)
            
        for metric in self.risk_metrics.metrics:
            for label, reqs in metric.get_requests().items():
                for req in reqs:
                    requests[label].add(req)

        return requests

    def compute_higher_derivatives(self):
        """Enable computation of second order derivatives."""
        self.requires_higher_order_derivatives=True

    def perform_prepocessing(self, request_interface : RequestInterface):
        """Perfrom preprocessing.
        
        Collect and Index requests for each product and exposure timepoint
        Perform rergression of products need to be built or exposure metrics need to be computated otherwise skip
        """
        request_interface.collect_and_index_requests(
            self.products,
            self.simulation_timeline,
            self._get_requests(),
            self.metric_exposure_timeline,
        )
        if self.requires_regression:
            self._perform_regression(request_interface)

    def _perform_regression(self, request_interface : RequestInterface):
        """Perform regression of each constructible product and expsoure timepoint."""
        # Set up Monte Carlo engine to simulate paths for pre-simulation
        engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_presim,
            num_steps=self.num_steps,
            is_pre_simulation=True
        )

        # Generate pre-simulation paths
        paths = engine.generate_paths()

        # Resolve all atomic and underlying requests
        resolved_requests = request_interface.resolve_requests(paths)

        for product in self.products:
            if self._product_requires_regression(product):
                self._perform_regression_for_product(product, resolved_requests)
            
    def _perform_regression_for_product(self, product : Product, resolved_requests : List[dict]):
        """
        Perform regression for a specific product via Dynamic Porgramming using the LSM algorithm
        """
        regression_timeline = set(product.regression_timeline.tolist()).union(self.exposure_timeline.tolist())
        regression_timeline = torch.tensor(sorted(regression_timeline), dtype=FLOAT, device=device)

        product_timeline = product.product_timeline
        product_regression_timeline = product.regression_timeline
        num_states = product.get_num_states()
        num_paths = self.num_paths_presim
        state_dtype = product.get_state_dtype()

        # Store date at which the last product cashflow has been computed (in reverse order)
        # starting at the last timepoint on the product timeline
        last_cf_index_computed = len(product_timeline)

        # Store last computed cashflows for each product state
        cf_cache = defaultdict(lambda: torch.zeros((num_paths, num_states), device=device))
        cf_cache[last_cf_index_computed] = torch.zeros((num_paths, num_states), device=device)

        # Perform regression via backwards induction and applying the LSM algorithm at each timepoint
        for reg_idx, t_reg in reversed(list(enumerate(regression_timeline))):
            total_cfs = torch.zeros((num_paths, num_states), device=device)

            product_time_idx = int(torch.searchsorted(product_timeline, t_reg).item())

            if product_time_idx >= len(product_timeline):
                continue
            t_next_idx = product_time_idx + 1 if product_timeline[product_time_idx] == t_reg else product_time_idx

            if t_next_idx < last_cf_index_computed:
                state_transition_matrix = torch.arange(
                    num_states, device=device, dtype=state_dtype
                ).expand(num_paths, num_states).clone()              

                step_value = torch.zeros((num_paths, num_states), device=device)

                # roll forward only the uncached window
                for idx in range(t_next_idx, last_cf_index_computed):
                    state_transition_matrix, cfs_matrix = product.compute_normalized_cashflows(
                        idx,
                        self.model,
                        resolved_requests,
                        self.regression_function,
                        state_transition_matrix
                    )
                    step_value += cfs_matrix  

                # stitch in cached tail from last_cf_index_computed
                tail_value = product.lookup_state_values(
                    cf_cache[last_cf_index_computed],
                    state_transition_matrix,
                )

                total_cfs = step_value + tail_value

                cf_cache[t_next_idx] = total_cfs.clone()
                last_cf_index_computed = t_next_idx

            else:
                total_cfs = cf_cache[t_next_idx]


            if t_reg in product_regression_timeline:
                i_t = product_timeline.tolist().index(t_reg)
                numeraire = resolved_requests[0][product.numeraire_requests[i_t].handle]
                explanatory = resolved_requests[0][product.spot_requests[(i_t, product.asset_ids[0])].handle]
            else:
                time_key = float(t_reg.item())
                exp_reg_idx = self._exposure_time_to_idx[time_key]
                numeraire = resolved_requests[0][self.numeraire_requests[(time_key, "numeraire")].handle]
                explanatory = resolved_requests[0][self.spot_requests[(time_key, product.asset_ids[0])].handle]

            normalized_cfs = numeraire.unsqueeze(1) * total_cfs

            A = self.regression_function.get_regression_matrix(explanatory)

            # Solve all states in one least-squares call
            solution = torch.linalg.lstsq(A, normalized_cfs).solution                        
            coeffs_mat = solution.transpose(0, 1).contiguous()                               

            # Store coeffs for this time in the product tensor and exposure tensor
            if t_reg in product_regression_timeline:
                product_reg_idx = torch.searchsorted(product_regression_timeline, t_reg).item()
                product.regression_coeffs[product_reg_idx, :, :] = coeffs_mat                

            if t_reg in self.exposure_timeline:
                exp_reg_idx = self._exposure_time_to_idx[float(t_reg.item())]
                self.regression_coeffs[product.product_id][exp_reg_idx, :, :] = coeffs_mat

    def _evaluate_product(self, product: Product, resolved_requests: List[dict]):
        num_paths = self.num_paths_mainsim
        initial_state = product.get_initial_state()
        state_transition_matrix = torch.full(
            (num_paths,),
            initial_state,
            dtype=product.get_state_dtype(),
            device=device,
        ).unsqueeze(1)

        exposures: list[torch.Tensor] = []
        t_start = 0
        cfs = torch.zeros(num_paths, dtype=FLOAT, device=device)

        # Case 1: no exposure timeline, only PV
        if not self.risk_metrics.requires_exposure_profiles() and self.risk_metrics.requires_discounted_cashflows():
            while t_start < len(product.product_timeline):
                state_transition_matrix, new_cfs = product.compute_normalized_cashflows(
                    t_start,
                    self.model,
                    resolved_requests,
                    self.regression_function,
                    state_transition_matrix
                )
                cfs += new_cfs[:, 0]
                t_start += 1

        else:
            # Case 2: exposures and maybe PV
            for i, t in enumerate(self.exposure_timeline):

                # advance product's realized state forward until exposure time point is reached
                while t_start < len(product.product_timeline) and product.product_timeline[t_start] <= t:
                    state_transition_matrix, new_cfs = product.compute_normalized_cashflows(
                        t_start,
                        self.model,
                        resolved_requests,
                        self.regression_function,
                        state_transition_matrix
                    )
                    cfs += new_cfs[:, 0]
                    t_start += 1

                time_key = float(t.item())
                numeraire = resolved_requests[0][self.numeraire_requests[(time_key, "numeraire")].handle]
                if self._can_use_analytic_exposure_for_product(product):
                    spot = resolved_requests[0][self.spot_requests[(time_key, product.asset_ids[0])].handle]
                    exposure = product.compute_discounted_exposure_analytically(
                        exposure_time=t,
                        spot=spot,
                        numeraire=numeraire,
                        model=self.model,
                    )
                else:
                    explanatory = resolved_requests[0][self.spot_requests[(time_key, product.asset_ids[0])].handle]
                    coeffs_all_states = self.regression_coeffs[product.product_id][i]
                    continuation = product.compute_continuation_values(
                        explanatory=explanatory,
                        regression_function=self.regression_function,
                        state_matrix=state_transition_matrix,
                        coeffs_all_states=coeffs_all_states,
                    ).squeeze(1)
                    exposure = continuation / numeraire                                   

                exposures.append(exposure)

            if self.risk_metrics.requires_discounted_cashflows():
                while t_start < len(product.product_timeline):
                    state_transition_matrix, new_cfs = product.compute_normalized_cashflows(
                        t_start,
                        self.model,
                        resolved_requests,
                        self.regression_function,
                        state_transition_matrix
                    )
                    cfs += new_cfs[:, 0]
                    t_start += 1

        exposures_tensor = (
            torch.stack(exposures, dim=0)
            if len(exposures) > 0
            else torch.zeros((0, num_paths), dtype=FLOAT, device=device)
        )
        return {
            PathwisePrimitive.DISCOUNTED_CASHFLOWS.value: cfs,
            PathwisePrimitive.EXPOSURE_PROFILES.value: exposures_tensor,
        }

    def _zero_metric_result(self, metric: Metric):
        num_evaluations = 1 if metric.metric_type in {MetricType.PV, MetricType.CVA, MetricType.EEPE} else len(self.metric_exposure_timeline)
        zero = torch.tensor(0.0, dtype=FLOAT, device=device)
        return [(zero, zero) for _ in range(num_evaluations)]

    def _initialize_analytical_metric_accumulators(self):
        zero = torch.tensor(0.0, dtype=FLOAT, device=device)
        return [
            [zero.clone() for _ in self.risk_metrics.metrics]
            for _ in self.netting_sets
        ]

    def _initialize_netting_set_accumulators(self):
        num_paths = self.num_paths_mainsim
        num_exposure_dates = len(self.exposure_timeline)
        accumulators = []
        for _ in self.netting_sets:
            accumulator = {}
            if self.risk_metrics.requires_primitive(PathwisePrimitive.DISCOUNTED_CASHFLOWS):
                accumulator[PathwisePrimitive.DISCOUNTED_CASHFLOWS.value] = torch.zeros(
                    num_paths,
                    dtype=FLOAT,
                    device=device,
                )
            if self.risk_metrics.requires_primitive(PathwisePrimitive.EXPOSURE_PROFILES):
                accumulator[PathwisePrimitive.EXPOSURE_PROFILES.value] = torch.zeros(
                    (num_exposure_dates, num_paths),
                    dtype=FLOAT,
                    device=device,
                )
            accumulators.append(accumulator)
        return accumulators

    def _evaluate_netting_set_from_accumulator(
        self,
        netting_set_idx: int,
        netting_set: NettingSet,
        netting_set_accumulator,
        resolved_requests: List[dict],
        analytical_metric_accumulator: list[torch.Tensor],
        has_pathwise_contribution: bool,
    ):
        if self.risk_metrics.requires_primitive(PathwisePrimitive.DISCOUNTED_CASHFLOWS):
            cfs = netting_set_accumulator[PathwisePrimitive.DISCOUNTED_CASHFLOWS.value]
        else:
            cfs = torch.zeros(self.num_paths_mainsim, dtype=FLOAT, device=device)

        if self.risk_metrics.requires_primitive(PathwisePrimitive.EXPOSURE_PROFILES):
            unsecured_exposures = netting_set.compute_unsecured_exposure_profiles(
                netted_exposures=netting_set_accumulator[PathwisePrimitive.EXPOSURE_PROFILES.value],
                exposure_timeline=self.exposure_timeline,
                metric_exposure_indices=self.metric_exposure_indices,
                delayed_exposure_indices=self.netting_set_delayed_exposure_indices[netting_set_idx],
            )
            exposure_list = [
                unsecured_exposures[idx]
                for idx in range(unsecured_exposures.shape[0])
            ]
        else:
            exposure_list = []

        metric_results = []
        for metric_idx, metric in enumerate(self.risk_metrics.metrics):
            if (
                metric.metric_type == MetricType.CVA
                and netting_set.counterparty_id is not None
                and getattr(metric, "counterparty_id", None) != netting_set.counterparty_id
            ):
                metric_results.append(self._zero_metric_result(metric))
                continue

            if metric.metric_type == MetricType.PV and metric.evaluation_type == Metric.EvaluationType.ANALYTICAL:
                analytical_value = analytical_metric_accumulator[metric_idx]
                if has_pathwise_contribution:
                    numerical_value, mc_error = metric._compute_mc_mean_and_error(cfs)
                else:
                    numerical_value = torch.zeros_like(analytical_value)
                    mc_error = torch.zeros_like(analytical_value)
                metric_results.append([(analytical_value + numerical_value, mc_error)])
                continue

            metric_results.append(
                metric.evaluate(
                    exposures=exposure_list,
                    cfs=cfs,
                    resolved_requests=resolved_requests,
                    netting_set=netting_set,
                    model=self.model,
                )
            )
        return metric_results

    def evaluate_products(self, resolved_requests : List[dict]) -> SimulationResults:
        """Compute metric outputs in main simulation phase."""
        netting_set_accumulators = self._initialize_netting_set_accumulators()
        analytical_metric_accumulators = self._initialize_analytical_metric_accumulators()
        has_pathwise_contribution = [False for _ in self.netting_sets]

        for product_idx, product in enumerate(self.products):
            netting_set_idx = self.product_to_netting_set_idx[product_idx]
            if self._can_skip_monte_carlo_for_product(product):
                for metric_idx, metric in enumerate(self.risk_metrics.metrics):
                    analytical_value = metric.evaluate_analytically(
                        product=product,
                        model=self.model,
                    )[0][0]
                    analytical_metric_accumulators[netting_set_idx][metric_idx] += analytical_value
                continue

            result = self._evaluate_product(product, resolved_requests)
            has_pathwise_contribution[netting_set_idx] = True
            if self.risk_metrics.requires_primitive(PathwisePrimitive.DISCOUNTED_CASHFLOWS):
                netting_set_accumulators[netting_set_idx][PathwisePrimitive.DISCOUNTED_CASHFLOWS.value] += result[
                    PathwisePrimitive.DISCOUNTED_CASHFLOWS.value
                ]
            if self.risk_metrics.requires_primitive(PathwisePrimitive.EXPOSURE_PROFILES):
                netting_set_accumulators[netting_set_idx][PathwisePrimitive.EXPOSURE_PROFILES.value] += result[
                    PathwisePrimitive.EXPOSURE_PROFILES.value
                ]

        results = []
        for idx, netting_set in enumerate(self.netting_sets):
            results.append(
                self._evaluate_netting_set_from_accumulator(
                    idx,
                    netting_set,
                    netting_set_accumulators[idx],
                    resolved_requests,
                    analytical_metric_accumulator=analytical_metric_accumulators[idx],
                    has_pathwise_contribution=has_pathwise_contribution[idx],
                )
            )

        # If differentiation is enabled, compute and store first order derivatives
        # via algorithmic adjoint differentiation (AAD) using PyTorch 'autograd' method
        # for each metric output for each product
        grads = []
        higher_grads = []
        if self.differentiate:
            model_params = self.model.get_model_params()
            for prod in results:
                grads_per_metric = []
                for metric in prod:
                    grads_per_eval = []
                    for eval,_ in metric:
                        _grads = torch.autograd.grad(
                            eval,
                            model_params,
                            retain_graph=True,
                            create_graph=self.requires_higher_order_derivatives,
                            allow_unused=True
                        )
                        grads_per_eval.append(_grads)
                    grads_per_metric.append(grads_per_eval)
                grads.append(grads_per_metric)

            # If higher order derivatives are enabled, the second order derivatives
            # are computed and stored
            if self.requires_higher_order_derivatives:
                for prod_grads in grads:
                    higher_grads_per_metric = []
                    for metric_grads in prod_grads:
                        evaluation_grads = []
                        for eval_grads in metric_grads:
                            higher_grads_per_eval = []
                            for grad in eval_grads:
                                _higher_grads = torch.autograd.grad(
                                    grad,
                                    model_params,
                                    retain_graph=True,
                                    allow_unused=True
                                )
                                higher_grads_per_eval.append(_higher_grads)
                            evaluation_grads.append(higher_grads_per_eval)
                        higher_grads_per_metric.append(evaluation_grads)
                    higher_grads.append(higher_grads_per_metric)

        netting_set_names = self._make_unique_names([netting_set.get_name() for netting_set in self.netting_sets])
        metric_names = self._make_unique_names([metric.get_name() for metric in self.risk_metrics.metrics])
        model_param_names = self.model.get_model_param_names()

        return SimulationResults(
            results,
            grads,
            higher_grads,
            netting_set_names=netting_set_names,
            metric_names=metric_names,
            model_param_names=model_param_names,
        )

    def run_simulation(self) -> SimulationResults:
        """Perform entire simulation to simulate metric outputs."""
        t0 = time.perf_counter()
        # Instatiate request interface to index, collect and solve requests by the model
        request_interface = RequestInterface(self.model)

        # Perform preprocessing:
        # - Collect and index requests for each product and exposure timepoint
        # - If products or exposure metrics need to be constructed a regression will be performed
        #   using the Longstaff-Schwartz algorithm 
        self.perform_prepocessing(request_interface)
        t1 = time.perf_counter()
        
        # Instantiate Monte Carlo engine to simulate paths in the main simulation phase
        main_engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_mainsim,
            num_steps=self.num_steps,
            is_pre_simulation=False
        )

        # Generate Monte Carlo paths
        paths=main_engine.generate_paths()
        t2 = time.perf_counter()

        # Resolve product and exposure requests 
        resolved_requests = request_interface.resolve_requests(paths)
        t3 = time.perf_counter()

        results = self.evaluate_products(resolved_requests)
        t4 = time.perf_counter()
        logger.info(
            "Simulation completed for %d netting set(s) and %d product(s): "
            "preprocessing=%.6fs path_generation=%.6fs request_resolution=%.6fs "
            "valuation=%.6fs total=%.6fs",
            len(self.netting_sets),
            len(self.products),
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t4 - t0,
        )

        return results
