# AutoregressiveConditionalBetas.jl

[![Build Status](https://github.com/Sicco123/AutoregressiveConditionalBetas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Sicco123/AutoregressiveConditionalBetas.jl/actions/workflows/CI.yml?query=branch%3Amain)

This code aims to replicate the autorgressive conditional beta algorithm presented in Blasques et al. (2024). The algorithm consists of a regression model with observation-driven time-varying paramaters. Both the regression parameters and the variance are defined by the score-driven updating equations. The model can be estimated with Quasi- Maximum Likelihood estimation. The model does not (yet) allow for extra regressors in the beta updating equations.

---

## Bibliography
Blasques, F., Francq, C., & Laurent, S. (2024). Autoregressive conditional betas. *Journal of Econometrics*, 238(2), 105630. ISSN 0304-4076. Available at [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304407623003469). DOI: [10.1016/j.jeconom.2023.105630](https://doi.org/10.1016/j.jeconom.2023.105630).
