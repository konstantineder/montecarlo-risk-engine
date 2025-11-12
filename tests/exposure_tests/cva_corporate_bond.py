from context import *
from helpers.cs_helper import CSHelper
import os
import sys
import pandas as pd
import numpy as np

from common.packages import *
from common.enums import SimulationScheme
import matplotlib.pyplot as plt
from controller.controller import SimulationController
from models.vasicek import VasicekModel
from models.cirpp import CIRPPModel
from models.model_config import ModelConfig
from metrics.cva_metric import CVAMetric 
from products.bond import Bond

sys.path.append(os.path.abspath(".."))
output_path = "tests/data/cds_data.csv"
df = pd.read_csv(output_path)
print(df.head())
#print(df["Company"].unique()


# Select 5Y tenor (most liquid)
TENOR_COL = "PX6"  

# Select 5 highly correlated names in same sector (banking, car manufacturers)
# NAMES = [
#     "JPMorgan Chase   Co",
#     "Bank of America Corp",
#     "Citigroup Inc",
#     "Wells Fargo   Co",
#     "Morgan Stanley",
# ]
NAMES = [
    "General Motors Co",
]

# Map column names to maturities (in years)
TENOR_MAP = {
    "PX1": 0.5, "PX2": 1.0, "PX3": 2.0, "PX4": 3.0, "PX5": 4.0,
    "PX6": 5.0, "PX7": 7.0, "PX8": 10.0, "PX9": 15.0, "PX10": 20.0
}
TENOR_COLS = list(TENOR_MAP.keys())
TENORS = [TENOR_MAP[c] for c in TENOR_COLS]

# Pick a common date that exists for all five
dates_per_name = {n: set(df.loc[df["Company"]==n, "Date"]) for n in NAMES}
common_dates = sorted(set.intersection(*dates_per_name.values()))
if not common_dates:
    raise RuntimeError("No common valuation date across the five names.")
# Pick first date as valuation date
valuation_date = common_dates[0] 

payment_days = np.arange(0.25, max(TENORS) + 0.0000001, 0.25)
discount_factors = np.exp(-0.02*payment_days)
recovery = 0.4

cs_helper = CSHelper()

# Display results and compare computed credit spreads based on
# bootstrapped hazard rates with data
def to_bps(x): return [round(1e4*v, 4) for v in x]

results = []
for name in NAMES:
    snap = df[(df["Company"] == name) & (df["Date"] == valuation_date)]
    row = snap.iloc[0]
    spreads = [float(row[c]) / 1e4 for c in TENOR_COLS]
    
    mats = TENORS
    haz = cs_helper.bootstrap_hazards(
        credit_spreads=spreads, 
        maturities=mats, 
        payment_days=payment_days, 
        discount_factors_payment_days=discount_factors, 
        recovery_rate=recovery
        )

    # Confirm: recompute model par spreads = Prot / RPV01 at each T_i
    check_spreads = []
    for i in range(len(mats)):
        prem, prot = cs_helper._compute_cds_legs(
            maturities=mats[: i+1],
            payment_days=payment_days,
            discount_factors_payment_days=discount_factors,
            recovery_rate=recovery,
            hazard_rates=haz[: i+1]
        )
        model_s = prot / prem
        check_spreads.append(model_s)
    
    results.append(pd.DataFrame({
        "Tenor (y)": TENORS,
        "Input spread (bps)": to_bps(check_spreads),
        "Model spread (bps)": to_bps(check_spreads),
        "Hazard λ (per year)": haz,
        "Company": name,
        "Date": valuation_date
    }))

check_df = pd.concat(results, ignore_index=True)
check_df["Abs error (bps)"] = (check_df["Model spread (bps)"] - check_df["Input spread (bps)"]).abs()
check_df.sort_values(["Company", "Tenor (y)"], inplace=True)
print(check_df[check_df["Company"]=="General Motors Co"].to_string(index=False))

hazards: dict[float, float] = dict(zip(TENORS, haz))
print(hazards)

# Setup model and product
interest_rate_model = VasicekModel(
    calibration_date=0.,
    rate=0.03,
    mean=0.05,
    mean_reversion_speed=0.02,
    volatility=0.2,
    asset_id="bond"
)

intensity_model = CIRPPModel(
    calibration_date=0.,
    y0=0.02,
    theta=0.05,
    kappa=0.02,
    volatility=0.02,
    hazard_rates=hazards,
    asset_id="default_intensity"
)
models = [interest_rate_model, intensity_model]
inter_correlation_matrices: list[np.ndarray] = []
inter_correlation_matrix = np.array([-0.9])
inter_correlation_matrices.append(inter_correlation_matrix)

model_config = ModelConfig(
    models=models,
    inter_asset_correlation_matrix=inter_correlation_matrix,
    credit_model_idx=1,
)

maturity = 2.0
zero_bond = Bond(
    startdate=0.0,
    maturity=2.0,
    notional=1,
    tenor=2,
    pays_notional=True, 
    fixed_rate=0.0,
    asset_id="bond"
)
portfolio=[zero_bond]

# Metric timeline for EE
exposure_timeline = np.linspace(0, 3.,100)
cva_metric = CVAMetric(recovery_rate=0.4)

metrics=[cva_metric]

num_paths_mainsim=10000
num_paths_presim=100000
num_steps=50
sc=SimulationController(portfolio, model_config, metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False, exposure_timeline)

sim_results=sc.run_simulation()

cva_bond=sim_results.get_results(0,0)[0]
print(cva_bond)


