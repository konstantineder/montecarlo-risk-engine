from products.product import *
from math import pi
from request_interface.request_types import AtomicRequestType, UnderlyingRequest, AtomicRequest
from collections import defaultdict

# Implementation of Equity
class Equity(Product):
    def __init__(self, asset_id: str | None = None):
        super().__init__(asset_ids=[asset_id])
        self.composite_req_handle=None

        asset_id = self.get_asset_id()
        self.spot_requests={(0, asset_id): AtomicRequest(AtomicRequestType.SPOT)}

    def __eq__(self, other):
        return isinstance(other, Equity) and self.get_asset_id() == other.get_asset_id()

    def __hash__(self):
        return hash(self.get_asset_id())
    
    def get_atomic_requests_for_underlying(self):
        requests=defaultdict(list)

        for label, req in self.spot_requests.items():
            requests[label].append(req)

        return requests
    
    def generate_underlying_requests_for_date(self, obervation_date):
        equity=Equity(self.get_asset_id())
        return UnderlyingRequest(equity)
    
    def get_value(self, resolved_atomic_requests):
        asset_id = self.get_asset_id()
        return self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_atomic_requests,
            request_type=AtomicRequestType.SPOT,
            time_idx=0,
            asset_id=asset_id,
        )
