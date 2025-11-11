from context import *
import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from controller.controller import SimulationController
from models.model_config import ModelConfig
from models.black_scholes import BlackScholesModel
from metrics.pv_metric import PVMetric
from products.basket_option import BasketOption, OptionType,BasketOptionType
from engine.engine import SimulationScheme


def test_pv_basket_option():
    # # --- GPU device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_assets = 4
    num_corrs = (num_assets-1)*num_assets//2

    inter_correlation_matrix = [[0.5] for _ in range(num_corrs)]
    inter_correlation_matrix = np.array(inter_correlation_matrix)

    #sigmas, corr = compute_sigmas_and_correlation_from_cholesky(L)
    models = []
    asset_ids = ["1","2","3","4"]
    for idx in range(4):
        models.append(BlackScholesModel(calibration_date=0.0,asset_id=asset_ids[idx],spot=100.0,rate=0.0,sigma=0.4))

    model=ModelConfig(models=models,inter_asset_correlation_matrix=inter_correlation_matrix)
    weights=[0.25,0.25,0.25,0.25]
    basket=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.ARITHMETIC,False)
    basket_geo=BasketOption(1.0,asset_ids,weights,100,OptionType.CALL,BasketOptionType.GEOMETRIC)

    portfolio=[basket,basket_geo]

    metrics = [PVMetric()]

    num_paths = 1000000
    steps = 1

    sc=SimulationController(portfolio=portfolio, 
                            model=model, 
                            metrics=metrics, 
                            num_paths_mainsim=num_paths, 
                            num_paths_presim=0, 
                            num_steps=steps, 
                            simulation_scheme=SimulationScheme.ANALYTICAL, 
                            differentiate=False)
    
    sim_results=sc.run_simulation()
    price_basket=sim_results.get_results(0,0)[0]
    price_geo=sim_results.get_results(1,0)[0]
    
    precision = 0.02
    assert abs(price_basket - 12.60) < precision
    assert abs(price_geo - 10.9551100513373) < precision