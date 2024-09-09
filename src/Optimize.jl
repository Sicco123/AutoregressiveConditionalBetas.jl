function stable_log(x)
    if x > 0
        return log(x)
    else
        #println("Warning: log(x) with x <= 0")
        return -999999
    end
end


function neg_loglik(params, T, y, X, g²_ν, g², μ,  X_dim, constant; warmup = 20)

    # Extract parameters by indexing directly into params
    ω_ν = params[1]
    α_ν = params[2]
    β_ν = params[3]
    ω̄ = view(params, 4:3+X_dim)
    ξ = view(params, 4+X_dim:3+2X_dim)
    c = view(params, 4+2X_dim:3+3X_dim)

    

    β_optim = zeros(size(X))

    # Call the prediction function
    ŷ, β_optim, g²_ν, ν̂ = predictions!(T, y, X, β_optim, ω_ν, α_ν, β_ν, g²_ν, μ, ω̄, ξ, c, g², constant)

    # Compute log-likelihood
    llik = sum(ν̂[warmup:end].^2 ./ g²_ν[warmup:end] .+ stable_log.(g²_ν[warmup:end]))/(T-warmup)

    return llik
end


function fit(model::ACB_model; optim_algorithm = BFGS(), g_tol = 0.000001, max_iter = 1000, max_iter_outer = 1000, show_trace = false, show_warnings = true)
    X_dim = size(model.X, 2)  
    T = size(model.y, 1)  

    # Flatten the parameters according to the lengths
    p0 = [model.ω_ν, model.α_ν, model.β_ν, model.ω̄..., model.ξ..., model.c...]

    # take exp of the parameters 
    p0[1:3] = log.(p0[1:3]) 
   
    # Optimizing with custom neg_loglik
    optm = Optim.optimize(
        p -> neg_loglik(vcat(exp.(p[1:3]), p[4:end]), T, model.y, model.X, model.g²_ν, model.g²,model.μ, X_dim, model.constant),
        p0,
        optim_algorithm, Optim.Options(g_tol=g_tol, iterations=max_iter, outer_iterations=max_iter_outer, show_trace=show_trace, show_warnings=show_warnings)
    )

    # Extract the optimized parameters
    optimized_params = Optim.minimizer(optm)
    optimized_params[1:3] = exp.(optimized_params[1:3])
    model.ω_ν, model.α_ν, model.β_ν, model.ω̄, model.ξ, model.c = optimized_params[1], optimized_params[2], optimized_params[3], view(optimized_params, 4:3+X_dim), view(optimized_params, 4+X_dim:3+2X_dim), view(optimized_params, 4+2X_dim:3+3X_dim)
 
     return optm
end
