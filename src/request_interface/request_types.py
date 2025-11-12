from __future__ import annotations
from enum import Enum 
from common.packages import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from products.product import Product

class AtomicRequestType(Enum):
    SPOT = 1
    DISCOUNT_FACTOR = 2
    NUMERAIRE = 3
    FORWARD_RATE = 4
    LIBOR_RATE = 5
    SURVIVAL_PROBABILITY = 6
    CONDITIONAL_SURVIVAL_PROBABILITY = 7

class AtomicRequest:
    def __init__(
        self, 
        request_type: AtomicRequestType,
        time1: float | None = None, 
        time2: float | None = None,
        id: int | None = None,
    ):
        self.request_type = request_type
        self.id=id
        self.time1 = time1
        self.time2 = time2
        self.handle: int | None = None  # Set during deduplication

    def set_handle(self, idx: int) -> None:
        self.handle=idx

    def key(self):
        return (self.request_type, self.id, self.time1, self.time2)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())
    
class UnderlyingRequest:
    def __init__(self, underlying_asset: Product):
        self.underlying_asset=underlying_asset

    def set_handle(self, idx: int):
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