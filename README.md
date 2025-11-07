Copyright (c) 2025 Dr. Konstantin K. C. Eder

# Monte Carlo Simulation Engine

A flexible, extensible Monte Carlo simulation engine for pricing and risk analytics of financial derivatives.

## Features

- **Metrics Calculation**
  - Present Value (**PV**)
  - Current Exposure (**CE**)
  - Expected Exposure (**EE**)
  - Effective Expected Exposure (**EEPE**)
  - Potential Future Exposure (**PFE**)
  - xVA metrics (e.g., CVA, DVA, FVA – under development)

- **Sensitivity Analysis**
  - Efficient and scalable **adjoint algorithmic differentiation (AAD)** using [PyTorch](https://pytorch.org/)
  - Payoff Smoothing via Fuzzy Logic to enable AAD for products with discontinuous payoffs (binary options, barrier options) 

- **Financial Products**
  - European Equity Options
  - European Bond Options
  - Binary Options
  - Bermudan Equity Options
  - Bermudan Bond Options
  - Bermudan Swaptions
  - American Options
  - FlexiCalls
  - Barrier Options
  - Basket Options
  - Interest Rate Swaps
  - Bonds (Zerobonds, Coupon Bonds, Floating Rate Notes)

- **Models**
  - **Black-Scholes Model**
  - **Black-Scholes Multi-asset Model**
  - Stochastic interest rate models:
    - **Vasicek**
    - **Hull-White**
  
## In Progress

- [ ] Extend the **request interface** to support composite requests
- [ ] Add **Libor Market Model (LMM)**
- [ ] Add **Merton** jump-diffusion model
- [x] Add credit derivatives (e.g. basket CDSs)
- [ ] Include netting sets and collateralization

## Architecture

- Object-oriented design based on:
  - Modular simulation controller
  - Metric interfaces
  - Regression-based continuation value estimation
  - Request-response model evaluation interface
- Regression and exposure calculation built on unified and extensible **timeline architecture**

## License

This codebase is available for **personal, non-commercial use only**. Please refer to the [LICENSE](LICENSE) file for full details.
