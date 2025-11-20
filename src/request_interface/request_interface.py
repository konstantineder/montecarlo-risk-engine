from enum import Enum 
from common.packages import *
from request_interface.request_types import AtomicRequest, AtomicRequestType
from collections import defaultdict
from products.product import Product
from models.model import Model

class RequestInterface:
    def __init__(self, model: Model):
        self.model = model # The model that can resolve requests
        self.num_requests=0
        self.all_atomic_requests=defaultdict(set)
        self.all_composite_requests=defaultdict(set)
    
    def _request_key(self, req: AtomicRequest) -> tuple[AtomicRequestType, float, float]:
        """Generate a hashable key for a request."""
        return (req.request_type, req.time1, req.time2)
    
    def collect_and_index_requests(
        self, 
        products: list[Product], 
        simulation_timeline: torch.Tensor, 
        exposure_requests: dict[tuple[int, str], list[AtomicRequest]], 
        exposure_timeline: torch.Tensor,
    ):
        """Collect, deduplicate per time point, and assign handles to all requests."""

        all_requests = defaultdict(set)
        all_comp_requests = defaultdict(set)
        time_to_index = {float(t): idx for idx, t in enumerate(simulation_timeline)}
        atomic_request_key_to_handle = {}  # key: (time_index, req) -> handle
        comp_request_key_to_handle = {}  # key: (time_index, req) -> handle

        atomic_counter = 0
        comp_counter = 0

        # From products
        for prod in products:
            underlying_requests = prod.get_underlying_requests()
            for und_time, und_reqs in underlying_requests.items():
                t = prod.modeling_timeline[und_time]
                time_idx=time_to_index[float(t)]
                for und_req in und_reqs:
                    all_comp_requests[time_idx].add(und_req)
                    comp_counter = self._register_underlying_request(
                        und_req, comp_request_key_to_handle, time_idx, comp_counter
                        )   
                    atomic_requests=und_req.get_atomic_requests()
                    for label, reqs in atomic_requests.items():
                        for req in reqs:
                            asset_id = label[1]
                            all_requests[(time_idx, asset_id)].add(req)
                            atomic_counter = self._register_atomic_request(
                                req, atomic_request_key_to_handle, time_idx, asset_id, atomic_counter
                                )

        # Collect atomic requests across products
        for prod in products:
            requests = prod.get_atomic_requests()
            for (t, asset_id), reqs in requests.items():
                time_index = time_to_index[float(prod.modeling_timeline[t])]
                for req in reqs:
                    label = (time_index, asset_id)
                    all_requests[label].add(req)
                    atomic_counter = self._register_atomic_request(
                        req, 
                        atomic_request_key_to_handle, 
                        time_index, 
                        asset_id, 
                        atomic_counter
                    )

        # Collect exposure requests from controller
        for (t, asset_id), exp_reqs in exposure_requests.items():
            time_index = time_to_index[float(exposure_timeline[t])]
            for exp_req in exp_reqs:
                label = (time_index, asset_id)
                all_requests[label].add(exp_req)
                atomic_counter = self._register_atomic_request(
                    exp_req, 
                    atomic_request_key_to_handle, 
                    time_index, 
                    asset_id, 
                    atomic_counter
                )

        self.all_requests = all_requests
        self.all_composite_requests = all_comp_requests

    def _register_atomic_request(self,req, request_key_to_handle, time_index, asset_id, counter):
        key = (time_index, asset_id, req)
        if key not in request_key_to_handle:
            request_key_to_handle[key] = counter
            counter += 1
        req.set_handle(request_key_to_handle[key])
        return counter
    
    def _register_underlying_request(self,req, request_key_to_handle, time_index, counter):
        key = (time_index, req)
        if key not in request_key_to_handle:
            request_key_to_handle[key] = counter
            counter += 1
        req.set_handle(request_key_to_handle[key])
        return counter

    def resolve_requests(self, paths: torch.Tensor):
        """Resolve all requests once paths have been simulated."""
        resolved_requests = {}
        resolved_composite_requests = {}

        for (t, asset_id), reqs in self.all_requests.items():
            for req in reqs:
                state = paths[:, t]
                result = self.model.resolve_request(req, asset_id, state)  # could be [num_paths] or [num_paths, num_assets]
                resolved_requests[req.handle] = result

        for _, comp_reqs in self.all_composite_requests.items():
            for req in comp_reqs:
                resolved_composite_requests[req.get_handle()] = req.get_value(resolved_requests)

        return [resolved_requests, resolved_composite_requests]
