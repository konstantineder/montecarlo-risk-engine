from enum import Enum 
from common.packages import *
from collections import defaultdict

class AtomicRequestType(Enum):
    SPOT=1
    DISCOUNT_FACTOR = 2
    NUMERAIRE = 3
    FORWARD_RATE = 4
    LIBOR_RATE = 5

class AtomicRequest:
    def __init__(self, request_type,time1=None, time2=None, id=None):
        self.request_type = request_type
        self.id=id
        self.time1 = time1
        self.time2 = time2
        self.handle = None  # Set during deduplication

    def set_handle(self, idx):
        self.handle=idx

    def key(self):
        return (self.request_type, self.id, self.time1, self.time2)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())
    
class CompositeRequest:
    def __init__(self,underlying_asset):
        self.underlying_asset=underlying_asset

    def set_handle(self, idx):
        self.underlying_asset.composite_req_handle=idx

    def get_handle(self):
        return self.underlying_asset.composite_req_handle

    def get_atomic_requests(self):
        return self.underlying_asset.get_atomic_requests_for_underlying()

    def get_value(self,resolved_atomic_requests):
        return self.underlying_asset.get_value(resolved_atomic_requests)
    
    def key(self):
        return (self.underlying_asset)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class RequestInterface:
    def __init__(self, model):
        self.model = model  # The model that can resolve requests
        self.num_requests=0
        self.all_atomic_requests=defaultdict(set)
        self.all_composite_requests=defaultdict(set)
        

    def _request_key(self, req):
        """Generate a hashable key for a request."""
        return (req.request_type, req.time1, req.time2)
    
    def collect_and_index_requests(self, products, simulation_timeline, exposure_requests, exposure_timeline):
        """Collect, deduplicate per time point, and assign handles to all requests."""
        from collections import defaultdict

        all_requests = defaultdict(set)
        all_comp_requests = defaultdict(set)
        time_to_index = {float(t): idx for idx, t in enumerate(simulation_timeline)}
        atomic_request_key_to_handle = {}  # key: (time_index, req) -> handle
        comp_request_key_to_handle = {}  # key: (time_index, req) -> handle

        atomic_counter = 0
        comp_counter = 0

        # From products
        for prod in products:
            composite_requests = prod.get_composite_requests()
            for comp_time, comp_reqs in composite_requests.items():
                t = prod.modeling_timeline[comp_time]
                time_idx=time_to_index[float(t)]
                for comp_req in comp_reqs:
                    all_comp_requests[time_idx].add(comp_req)
                    comp_counter = self._register(comp_req, comp_request_key_to_handle, time_idx, comp_counter)   
                    atomic_requests=comp_req.get_atomic_requests()
                    for _, reqs in atomic_requests.items():
                        for req in reqs:
                            all_requests[time_idx].add(req)
                            atomic_counter = self._register(req, atomic_request_key_to_handle,time_idx, atomic_counter)

        # From products
        for prod in products:
            requests = prod.get_atomic_requests()
            for (t, asset_id), reqs in requests.items():
                time_index = time_to_index[float(prod.modeling_timeline[t])]
                for req in reqs:
                    all_requests[(time_index, asset_id)].add(req)
                    atomic_counter = self._register(
                        req, 
                        atomic_request_key_to_handle, 
                        time_index, 
                        asset_id, 
                        atomic_counter
                    )

        # From external exposure requests
        for (t, asset_id), exp_reqs in exposure_requests.items():
            time_index = time_to_index[float(exposure_timeline[t])]
            for exp_req in exp_reqs:
                all_requests[(time_index, asset_id)].add(exp_req)
                atomic_counter = self._register(
                    exp_req, 
                    atomic_request_key_to_handle, 
                    time_index, 
                    asset_id, 
                    atomic_counter
                )

        self.all_requests = all_requests
        self.all_composite_requests = all_comp_requests

    def _register(self,req, request_key_to_handle, time_index, asset_id, counter):
        key = (time_index, asset_id, req)
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

        for (t, asset_id), comp_reqs in self.all_composite_requests.items():
            for req in comp_reqs:
                resolved_composite_requests[req.get_handle()] = req.get_value(resolved_requests)

        return [resolved_requests, resolved_composite_requests]
