from products.product import *
from math import pi
from request_interface.request_types import AtomicRequestType, UnderlyingRequest, AtomicRequest
from collections import defaultdict

class Bond(Product):
    """
    Bond implementation
    Supports both fixed and floating coupon payments
    Depending on product specification supports Zerobonds, Coupon bonds and floating rate notes (FRN)
    """
    def __init__(
        self, 
        startdate    : float, # Startdate of contract
        maturity     : float, # Maturity of contract
        notional     : float, # Principal/Notional
        tenor        : float, # Tenor of coupon payments
        pays_notional: Optional[bool]=True, # Pays notional at maturity (True by default)
        fixed_rate   : Optional[float]=None, # Specifiy fixed interest rate in case of fixed coupon payments (if none floating coupons are payed)
        asset_id     : str | None = None,
    ):
        
        super().__init__(asset_ids=[asset_id])
        self.startdate = torch.tensor([startdate], dtype=FLOAT,device=device)
        self.maturity = torch.tensor([maturity], dtype=FLOAT,device=device)
        self.notional = torch.tensor([notional], dtype=FLOAT, device=device)
        self.tenor = torch.tensor([tenor], dtype=FLOAT, device=device)
        self.fixed_rate = fixed_rate
        self.pays_notional=pays_notional
        self.composite_req_handle=None

        self.payment_dates = []

        # Build the payment schedule and requests
        date = startdate + tenor
        idx = 0
        
        asset_id = self.get_asset_id()
        self.atomic_requests_for_underlying: dict[tuple[int, str | None], AtomicRequest] = {}
        if fixed_rate is not None:
            while date < maturity:
                self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, date)
                self.atomic_requests_for_underlying[(idx, asset_id)] = AtomicRequest(AtomicRequestType.FORWARD_RATE,startdate,date)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, maturity)
            self.atomic_requests_for_underlying[(idx, asset_id)] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=maturity)
            self.payment_dates.append(maturity)
        else:
            while date < maturity:
                self.libor_requests[(idx, asset_id)] = AtomicRequest(AtomicRequestType.LIBOR_RATE, date - tenor, date)
                self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, date)
                self.atomic_requests_for_underlying[(idx, asset_id)] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.libor_requests[(idx, asset_id)] = AtomicRequest(AtomicRequestType.LIBOR_RATE, date - tenor, maturity)
            self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, maturity)
            self.atomic_requests_for_underlying[(idx, asset_id)] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
            self.atomic_requests_for_underlying[(idx + 1, asset_id)] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=maturity)
            self.payment_dates.append(maturity)

        self.payment_dates = torch.tensor(self.payment_dates, dtype=FLOAT, device=device)
        self.product_timeline = self.payment_dates
        self.modeling_timeline = self.payment_dates
        self.regression_timeline = torch.tensor([], dtype=FLOAT, device=device)

    def __eq__(self, other):
        return (
            isinstance(other, Bond) and
            torch.equal(self.startdate, other.startdate) and
            torch.equal(self.maturity, other.maturity) and
            torch.equal(self.tenor, other.tenor) and
            self.fixed_rate == other.fixed_rate and
            self.pays_notional == other.pays_notional
        )

    def __hash__(self):
        return hash((
            self.startdate.item(),
            self.maturity.item(),
            self.tenor.item(),
            self.fixed_rate,
            self.pays_notional
        ))
    
    def get_atomic_requests_for_underlying(self):
        requests=defaultdict(list)

        for label, req in self.atomic_requests_for_underlying.items():
            requests[label].append(req)

        return requests
    
    def generate_underlying_requests_for_date(self, observation_date):
        bond = Bond(
            observation_date,
            self.maturity.item(),
            self.notional.item(),
            self.tenor.item(),
            self.pays_notional,
            self.fixed_rate,
            asset_id=self.get_asset_id(),
            )
        
        return UnderlyingRequest(bond)
    
    def get_value(self, resolved_atomic_requests):
        if self.fixed_rate is not None:
            return self.get_value_fixed(resolved_atomic_requests)
        else:
            return self.get_value_float(resolved_atomic_requests)

    def get_value_fixed(self, resolved_atomic_requests):
        total_cashflow = torch.zeros_like(resolved_atomic_requests[0], dtype=FLOAT, device=device)

        prev_time=self.startdate
        asset_id = self.get_asset_id()
        for t in self.numeraire_requests.keys():
            discount_req = self.atomic_requests_for_underlying[(t, asset_id)]

            discount = resolved_atomic_requests[discount_req.handle]

            time=self.modeling_timeline[t]
            dt=time - prev_time

            total_cashflow += self.notional * self.fixed_rate * dt * discount

            prev_time=time
        
        if self.pays_notional:
            discount_req = self.atomic_requests_for_underlying[(len(self.modeling_timeline) - 1, asset_id)]
            discount = resolved_atomic_requests[discount_req.handle]
            total_cashflow += self.notional * discount

        return total_cashflow
    
    def get_value_float(self, resolved_atomic_requests: dict[tuple[int, str], list[torch.Tensor]]) -> torch.Tensor:
        total_cashflow = torch.zeros_like(resolved_atomic_requests[0], dtype=FLOAT, device=device)

        asset_id = self.get_asset_id()
        for t in self.numeraire_requests.keys():
            discount_req = self.atomic_requests_for_underlying[(t, asset_id)]
            discount_next_req = self.atomic_requests_for_underlying[(t+1, asset_id)]

            discount = resolved_atomic_requests[discount_req.handle]
            discount_next = resolved_atomic_requests[discount_next_req.handle]

            total_cashflow += self.notional * (discount - discount_next)

        if self.pays_notional:
            discount_req = self.atomic_requests_for_underlying[(len(self.modeling_timeline) - 1, asset_id)]
            discount = resolved_atomic_requests[discount_req.handle]
            total_cashflow += self.notional * discount

        return total_cashflow
    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests,regression_RegressionFunction=None, state=None):
        if self.fixed_rate is not None:
            return self.compute_normalized_cashflows_fixed(time_idx, model, resolved_requests,regression_RegressionFunction, state)
        else:
            return self.compute_normalized_cashflows_float(time_idx, model, resolved_requests,regression_RegressionFunction, state)
    
    def compute_normalized_cashflows_fixed(self, time_idx, model, resolved_requests,regression_RegressionFunction=None, state=None):
        numeraire= resolved_requests[0][self.numeraire_requests[time_idx].handle]

        prev_time=self.startdate
        if time_idx > 0:
            prev_time = self.payment_dates[time_idx - 1]

        dt=self.payment_dates[time_idx] - prev_time

        cashflow = self.fixed_rate * dt

        if self.pays_notional and time_idx == len(self.modeling_timeline) - 1:
            cashflow+=self.notional

        discounted_cashflow = cashflow / numeraire
        return state, discounted_cashflow.unsqueeze(1)
    
    def compute_normalized_cashflows_float(self, time_idx, model, resolved_requests,regression_RegressionFunction=None, state=None):
        libor_rate = self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_requests[0],
            request_type=AtomicRequestType.LIBOR_RATE,
            time_idx=time_idx,
            asset_id=self.get_asset_id()
        )
        
        numeraire = self.get_resolved_atomic_request(
            resolved_atomic_requests=resolved_requests[0],
            request_type=AtomicRequestType.NUMERAIRE,
            time_idx=time_idx,
        )

        prev_time=self.startdate
        if time_idx > 0:
            prev_time = self.payment_dates[time_idx - 1]

        dt=self.payment_dates[time_idx] - prev_time

        cashflow = libor_rate * dt

        if self.pays_notional and time_idx == len(self.modeling_timeline):
            cashflow+=self.notional

        discounted_cashflow = cashflow / numeraire
        return state, discounted_cashflow.unsqueeze(1)
