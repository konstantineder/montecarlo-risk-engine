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
from products.swap import InterestRateSwap, IRSType

sys.path.append(os.path.abspath(".."))
output_path = "tests/data/cds_data.csv"
df = pd.read_csv(output_path)

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
#print(check_df[check_df["Company"]=="General Motors Co"].to_string(index=False))

hazards: dict[float, float] = dict(zip(TENORS, haz))
#print(hazards)
def test_cva_corporate_bond():
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
    inter_correlation_matrix = np.array([0.0])
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
    
    pv_bond = interest_rate_model.compute_bond_price(0.0, maturity, 0.03)
    survival_prob = intensity_model.survival_probability(0.0, maturity, 0.0001)
    expected_loss = (1 - recovery) * (1 - survival_prob) * pv_bond
    
    assert abs(cva_bond.item() - expected_loss.item()) < 2e-6
    
def test_cva_wwr_payer_swap():
    # Setup model and product
    interest_rate_model = VasicekModel(
        calibration_date=0.,
        rate=0.03,
        mean=0.05,
        mean_reversion_speed=0.02,
        volatility=0.2,
        asset_id="irs"
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
    inter_correlation_matrix = np.array([0.99999])
    inter_correlation_matrices.append(inter_correlation_matrix)

    model_config = ModelConfig(
        models=models,
        inter_asset_correlation_matrix=inter_correlation_matrix,
    )

    maturity = 10.0
    irs = InterestRateSwap(
        startdate=0.0,
        enddate=maturity,
        notional=1.0,
        fixed_rate=0.03,
        tenor_fixed=0.25,
        tenor_float=0.25, 
        irs_type=IRSType.PAYER,
        asset_id="irs"
    )
    portfolio=[irs]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, maturity,100)
    cva_metric = CVAMetric(counterparty_id=counterparty_id,recovery_rate=0.4)

    metrics=[cva_metric]
    risk_metrics=RiskMetrics(metrics=metrics, exposure_timeline=exposure_timeline)

    num_paths_mainsim=100000
    num_paths_presim=100000
    num_steps=10
    sc=SimulationController(portfolio, model_config, risk_metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False)
    monte_carlo_err=1.96*1/np.sqrt(num_paths_mainsim)
    sim_results=sc.run_simulation()

    cva_irs=sim_results.get_results(0,0)[0]

    cva_uncorr=1.1145761564845404  # from test without WWR

    assert cva_irs.item() - cva_uncorr > 0.025  # CVA with WWR should be larger than without WWR
