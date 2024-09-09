module AutoregressiveConditionalBetas

# Write your package code here.

# Place all your imports here
using Distributions
using Random
using LinearAlgebra
using Distributed
using Optim
using ARCHModels

include("Model.jl")
include("utils/GARCH.jl")
include("Predict.jl")
include("Optimize.jl")

# Include all your functions and constants here
export 
# ACB 
       ACB_model
       
# utils/GARCH.jl
       GARCH_volatilities
# Predict.jl
       predict_t!
       update_βi
       predictions
       predictions!
# Optimize.jl 
       fit 
       neg_loglik

end #
