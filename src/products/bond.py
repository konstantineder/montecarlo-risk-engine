from products.product import *
from math import pi
from request_interface.request_interface import AtomicRequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# Bond implementation
# Supports both fixed and floating coupon payments
# Depending on product specification supports Zerobonds, Coupon bonds and floating rate notes (FRN)
class Bond(Product):
    def __init__(self, 
                 startdate    : float, # Startdate of contract
                 maturity     : float, # Maturity of contract
                 notional     : float, # Principal/Notional
                 tenor        : float, # Tenor of coupon payments
                 pays_notional: Optional[bool]=True, # Pays notional at maturity (True by default)
                 fixed_rate   : Optional[float]=None # Specifiy fixed interest rate in case of fixed coupon payments (if none floating coupons are payed)
                 ):
        
        super().__init__()
        self.startdate = torch.tensor([startdate], dtype=FLOAT,device=device)
        self.maturity = torch.tensor([maturity], dtype=FLOAT,device=device)
        self.notional = torch.tensor([notional], dtype=FLOAT, device=device)
        self.tenor = torch.tensor([tenor], dtype=FLOAT, device=device)
        self.fixed_rate = fixed_rate
        self.pays_notional=pays_notional
        self.composite_req_handle=None

        self.payment_dates = []
        self.libor_requests = {}   
        self.underlying_requests = {}
        self.numeraire_requests = {}  

        # Build the payment schedule and requests
        date = startdate + tenor
        idx = 0

        if fixed_rate is not None:
            while date < maturity:
                self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, date)
                self.underlying_requests[idx] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=date)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.DISCOUNT_FACTOR, maturity)
            self.underlying_requests[idx] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=maturity)
            self.payment_dates.append(maturity)
        else:
            while date < maturity:
                self.libor_requests[idx] = AtomicRequest(AtomicRequestType.LIBOR_RATE, date - tenor, date)
                self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.NUMERAIRE, date)
                self.underlying_requests[idx] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.libor_requests[idx] = AtomicRequest(AtomicRequestType.LIBOR_RATE, date - tenor, maturity)
            self.numeraire_requests[idx] = AtomicRequest(AtomicRequestType.DISCOUNT_FACTOR, maturity)
            self.underlying_requests[idx] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
            self.underlying_requests[idx + 1] = AtomicRequest(request_type=AtomicRequestType.FORWARD_RATE,time1=startdate,time2=maturity)
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
    
    def get_atomic_requests(self):
        requests=defaultdict(list)

        for t, req in self.numeraire_requests.items():
            requests[t].append(req)

        if self.fixed_rate is None:
            for t, req in self.libor_requests.items():
                requests[t].append(req)

        return requests
    
    def get_atomic_requests_for_underlying(self):
        requests=defaultdict(list)

        for t, req in self.underlying_requests.items():
            requests[t].append(req)

        return requests
    
    def generate_composite_requests_for_date(self, observation_date):
        bond = Bond(observation_date,self.maturity.item(),self.notional.item(),self.tenor.item(),self.pays_notional,self.fixed_rate)
        return CompositeRequest(bond)
    
    def get_value(self, resolved_atomic_requests):
        if self.fixed_rate is not None:
            return self.get_value_fixed(resolved_atomic_requests)
        else:
            return self.get_value_float(resolved_atomic_requests)

    def get_value_fixed(self, resolved_atomic_requests):
        total_cashflow = torch.zeros_like(resolved_atomic_requests[0], dtype=FLOAT, device=device)

        prev_time=self.startdate
        for t in self.numeraire_requests.keys():
            discount_req = self.underlying_requests[t]

            discount = resolved_atomic_requests[discount_req.handle]

            time=self.modeling_timeline[t]
            dt=time - prev_time

            total_cashflow += self.notional * self.fixed_rate * dt * discount

            prev_time=time
        
        if self.pays_notional:
            discount_req = self.underlying_requests[len(self.modeling_timeline) - 1]
            discount = resolved_atomic_requests[discount_req.handle]
            total_cashflow += self.notional * discount

        return total_cashflow
    
    def get_value_float(self, resolved_atomic_requests):
        total_cashflow = torch.zeros_like(resolved_atomic_requests[0], dtype=FLOAT, device=device)

        for t in self.numeraire_requests.keys():
            discount_req = self.underlying_requests[t]
            discount_next_req = self.underlying_requests[t+1]

            discount = resolved_atomic_requests[discount_req.handle]
            discount_next = resolved_atomic_requests[discount_next_req.handle]

            total_cashflow += self.notional * (discount - discount_next)

        if self.pays_notional:
            discount_req = self.underlying_requests[len(self.modeling_timeline) - 1]
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

        if self.pays_notional and time_idx == len(self.modeling_timeline):
            cashflow+=self.notional

        discounted_cashflow = cashflow / numeraire
        return state, discounted_cashflow.unsqueeze(1)
    
    def compute_normalized_cashflows_float(self, time_idx, model, resolved_requests,regression_RegressionFunction=None, state=None):
        libor_rate = resolved_requests[0][self.libor_requests[time_idx].handle]
        numeraire= resolved_requests[0][self.numeraire_requests[time_idx].handle]

        prev_time=self.startdate
        if time_idx > 0:
            prev_time = self.payment_dates[time_idx - 1]

        dt=self.payment_dates[time_idx] - prev_time

        cashflow = libor_rate * dt

        if self.pays_notional and time_idx == len(self.modeling_timeline):
            cashflow+=self.notional

        discounted_cashflow = cashflow / numeraire
        return state, discounted_cashflow.unsqueeze(1)
