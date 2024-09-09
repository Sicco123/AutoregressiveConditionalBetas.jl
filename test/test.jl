using AutoregressiveConditionalBetas
using Random
using CSV
using DataFrames
using Plots
using ARCHModels

function load_factor_data()
    # Load the CSV file
    data_1 = CSV.read("test/data/agg_HFMarket.csv", DataFrame, header=true)
    data_2 = CSV.read("test/data/agg_HFSMB.csv", DataFrame, header=true)
    data_3 = CSV.read("test/data/agg_HFHML.csv", DataFrame, header=true)

    # take Column3Sum from each data and put together in matrix
    data = hcat(data_1.Column3Sum, data_2.Column3Sum, data_3.Column3Sum)
    data .= data .* 100
    
    return data
end

function simulate_GARCH_data(N; warmup = 500)
    ϵ = randn(N+warmup)
    initial_σ = 0.25
    σ = zeros(N+warmup)
    σ[1] = initial_σ
    for i in 2:N+warmup
        σ[i] = 0.005 +  0.05*ϵ[i-1]^2 + 0.94*σ[i-1]
        ϵ[i] = ϵ[i] * sqrt(σ[i])
    end
    
    ϵ = ϵ[warmup+1:end]
    σ = σ[warmup+1:end]


    errors = ϵ
    volatility_values = σ


    llik = sum(errors[20:end].^2 ./ volatility_values[19:end-1] .+ log.(volatility_values[19:end-1]))

    # println("Mean: ", mean(errors[20:end]))
    # println("Mean: ", mean(volatility_values[2019:end-1]))

    # println(volatility_values[1:10])
    # println(volatility_values[end-10:end])
    # println(errors[1:10])
    # println(errors[end-10:end])
    println("Log-likelihood: ", llik/(N-20))
    return errors
end

function main()
    # generate normal data
    Random.seed!(44)

    N = 4000
    y = simulate_GARCH_data(N)
    #println(y)
    X = load_factor_data()
    X = X[1:N, :]
    # Add a constant
    X = hcat(ones(size(X, 1)), X)
    b =  [0.001, 1.0, 0.5, 0.25]

    y .+=  X*b  



    model = nothing 

    params = nothing 
    params = [0.005, 0.05, 0.94, 0.001, 0.06, 0.04, 0.02, 0.05, 0.05, 0.05, 0.05, 0.94, 0.94, 0.94, 0.94]
    
    model = ACB_model(
        y,
        X, 
        params[1],  # ω_ν
        params[2],  # α_ν
        params[3],  # β_ν
        params[4:7],  # ω̄
        params[8:11],    # ξ
        params[12:15],    # c
        [[0.0], [0.0], [0.0], [0.0]]; # γ
        constant = true 
    )





    res = AutoregressiveConditionalBetas.fit(model; show_trace=true)

    println("Optimization result:")
    println(res)
    

    println("Optimized parameters:")
    println("ω_ν: ", model.ω_ν)
    println("α_ν: ", model.α_ν)
    println("β_ν: ", model.β_ν)
    println("μ: ", model.μ)
    println("ω̄: ", model.ω̄)
    println("ξ: ", model.ξ)
    println("c: ", model.c)
    

    y_hat, β =  AutoregressiveConditionalBetas.predictions(model)



    # llik = AutoregressiveConditionalBetas.neg_loglik(params, N, y, X, model.g²_ν, model.g², model.μ, 4, true)
    #println("Log-likelihood: ", llik)

    # println(round.(model.g²[:,4], digits=3))
    # println(round.(model.β, digits=3))

    # plot y hat 

    # plot y 
    plot(y, title="y", label="y")
    plot!(y_hat, title="y_hat", label="y_hat")
    savefig("test/figures/y_hat.pdf")

    # plot the second third and fourth model.β in separate plots
    plot(model.β[1:1000, 1], title="First Model β")
    savefig("test/figures/ACB_model_1.png")

    plot(model.β[1:1000, 2], title="Second Model β")
    savefig("test/figures/ACB_model_2.png")

    plot(model.β[1:1000, 3], title="Third Model β")
    savefig("test/figures/ACB_model_3.png")

    plot(model.β[1:1000, 4], title="Fourth Model β")
    savefig("test/figures/ACB_model_4.png")
    
    # # save 


end

# Run the main function
main()
