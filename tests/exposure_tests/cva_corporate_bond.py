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
from metrics.risk_metrics import RiskMetrics
from products.bond import Bond
from helpers.kaggle_data_helper import download_and_retrieve_data_from_kaggle

HANDLE = "debashish311601/credit-default-swap-cds-prices"

df = download_and_retrieve_data_from_kaggle(handle=HANDLE, relative_output_path="cds_data.csv")

# Select 5Y tenor (most liquid)
TENOR_COL = "PX6"  

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
#print(check_df[check_df["Company"]=="General Motors Co"].to_string(index=False))

hazards: dict[float, float] = dict(zip(TENORS, haz))

def compute_cva_zero_bond(correlation: float):
    """Compute CVA of zero-coupon bond with given correlation between interest rate 
    and counterparty intensity.
    """

    # Setup model and product
    interest_rate_model = VasicekModel(
        calibration_date=0.,
        rate=0.03,
        mean=0.05,
        mean_reversion_speed=1,
        volatility=0.2,
        asset_id="bond"
    )
    counterparty_id = "General Motors Co"
    intensity_model = CIRPPModel(
        calibration_date=0.,
        y0=0.0001,
        theta=0.01,
        kappa=0.1,
        volatility=0.02,
        hazard_rates=hazards,
        asset_id=counterparty_id
    )
    models = [interest_rate_model, intensity_model]
    inter_correlation_matrices: list[np.ndarray] = []
    inter_correlation_matrix = np.array([correlation])
    inter_correlation_matrices.append(inter_correlation_matrix)

    model_config = ModelConfig(
        models=models,
        inter_asset_correlation_matrix=inter_correlation_matrix,
    )

    maturity = 2.0
    zero_bond = Bond(
        startdate=0.0,
        maturity=maturity,
        notional=1,
        tenor=maturity,
        pays_notional=True, 
        fixed_rate=0.0,
        asset_id="bond"
    )
    portfolio=[zero_bond]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, maturity,100)
    cva_metric = CVAMetric(counterparty_id=counterparty_id, recovery_rate=0.4)
    risk_metrics = RiskMetrics(metrics=[cva_metric], exposure_timeline=exposure_timeline)

    num_paths_mainsim=100000
    num_paths_presim=100000
    num_steps=10
    sc=SimulationController(
        portfolio=portfolio, 
        model=model_config, 
        risk_metrics=risk_metrics, 
        num_paths_mainsim=num_paths_mainsim, 
        num_paths_presim=num_paths_presim, 
        num_steps=num_steps, 
        simulation_scheme=SimulationScheme.EULER, 
        differentiate=False,
    )

    sim_results=sc.run_simulation()

    cva_bond=sim_results.get_results(0,0)[0]
    cva_bond_error=sim_results.get_mc_error(0,0)[0]
    
    return (cva_bond, cva_bond_error)
    
correlations = np.linspace(-0.95, 0.95, 25)

cva_vals = []
cva_errs = []

for rho in correlations:
    cva, err = compute_cva_zero_bond(rho)
    cva_vals.append(cva)
    cva_errs.append(err)

plt.figure(figsize=(8,5))

# error bar plot
plt.errorbar(
    correlations, cva_vals, yerr=cva_errs,
    fmt='o-', color='black', ecolor='gray', capsize=3, label='CVA with correlation'
)

cva_uncorr, cva_uncorr_err = compute_cva_zero_bond(0.0)

# baseline uncorrelated CVA
plt.axhline(
    y=cva_uncorr, color='blue', linestyle='--',
    label=f'Uncorrelated CVA = {cva_uncorr:.4f}'
)

plt.xlabel("Correlation ρ between interest rate and intensity")
plt.ylabel("CVA")
plt.title("CVA vs Correlation (Wrong-Way / Right-Way Risk)")
plt.grid(True)
plt.legend()

out_dir = os.path.join("tests", "plots", "exposure_tests")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "cva_bond.png")
plt.savefig(out_path)
print(f"Plot saved to {out_path}")




