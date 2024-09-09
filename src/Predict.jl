function predict_t!( xₜ::Vector{Float64}, βₜ::Vector{Float64})
    yₜ = dot(βₜ, xₜ)
    return yₜ
end


# Update β_i for each parameter
function update_βi( ω̄ᵢ::Float64, ξᵢ::Float64, νₜ::Float64, xᵢₜ::Float64, μₜᵢ::Float64, g²ᵢₜ::Float64, cᵢ::Float64, βᵢₜ::Float64, γᵢ::Vector{Float64}, zₜ::Vector{Float64}, _::Bool)
    βᵢₜ₊₁ = ω̄ᵢ + ξᵢ * (νₜ * xᵢₜ / (μₜᵢ^2 + g²ᵢₜ)) + cᵢ * βᵢₜ + dot(γᵢ, zₜ)
    return βᵢₜ₊₁
end

function update_βi( ω̄ᵢ::Float64, ξᵢ::Float64, νₜ::Float64, xᵢₜ::Float64, μₜᵢ::Float64, g²ᵢₜ::Float64, cᵢ::Float64, βᵢₜ::Float64, _::Bool)
    βᵢₜ₊₁ = ω̄ᵢ + ξᵢ * (νₜ * xᵢₜ / (μₜᵢ^2 + g²ᵢₜ)) + cᵢ * βᵢₜ 

    return βᵢₜ₊₁
end


function update_βi_constant( ω̄ᵢ::Float64, ξᵢ::Float64, νₜ::Float64, xᵢₜ::Float64, μₜᵢ::Float64, g²ᵢₜ::Float64, cᵢ::Float64, βᵢₜ::Float64, constant::Bool)
    βᵢₜ₊₁ = ω̄ᵢ + ξᵢ * (νₜ * xᵢₜ ) + cᵢ * βᵢₜ 
    return βᵢₜ₊₁
end

function update_g²_ν(ω_ν::Float64, α_ν::Float64, β_ν::Float64, g²_νₜ₋₁::Float64, νₜ₋₁::Float64)

    
    g²_νₜ = ω_ν + α_ν * νₜ₋₁^2 + β_ν * g²_νₜ₋₁
    return g²_νₜ
end

function predictions(model::ACB_model)
    T = size(model.y)[1]
    y = model.y
    X = model.X
    β = model.β
    ω_ν = model.ω_ν
    α_ν = model.α_ν
    β_ν = model.β_ν
    g²_ν = model.g²_ν
    μ = model.μ
    ω̄ = model.ω̄
    ξ = model.ξ
    c = model.c
    g² = model.g²
    constant = model.constant

    
    ŷ, β, g²_ν, ν̂  = predictions!(T, y, X, β, ω_ν, α_ν, β_ν, g²_ν, μ, ω̄, ξ, c, g², constant)

    model.β = β
    model.g²_ν = g²_ν
    model.g² = g²


    return ŷ, β
end



function predictions!(T, y, X, β, ω_ν, α_ν, β_ν, g²_ν, μ, ω̄, ξ, c, g², constant)  

    ŷ = zeros(T)
    ν̂ = zeros(T)

    # In case constant is true we use a different updating rule for the first β
    update_β_place_holder = constant ? update_βi_constant : update_βi

    # initialize first β
    # initial_values = ω̄ ./ (1 .- c)
    # β[1 , :] = initial_values
    # β[2 , :] = initial_values

    # initialize first g²_ν
    # initial_value = ω_ν / (1 - α_ν - β_ν)
    # g²_ν[1] = initial_value
    # g²_ν[2] = initial_value
     

    for t in 2:T-1
        ŷ[t] =  predict_t!(X[t, :], β[t,:])
        ν̂[t] = y[t] - ŷ[t]
        
        
        # compute next g²_ν
        g²_ν[t+1] = update_g²_ν(ω_ν, α_ν, β_ν, g²_ν[t], ν̂[t])


        # compute first β
        β[t + 1, 1] = update_β_place_holder(ω̄[1], ξ[1], ν̂[t], X[t, 1], μ[1], g²[t - 1, 1], c[1], β[t, 1], constant)

        # compute next βs    
        β[t + 1, 2:end] .= update_βi.(ω̄[2:end], ξ[2:end], ν̂[t], X[t, 2:end], μ[2:end], g²[t - 1, 2:end], c[2:end], β[t, 2:end], constant)

    end    
    
    ŷ[T] = predict_t!(X[T, :], β[T,:])
    ν̂[T] = y[T] - ŷ[T]

    return ŷ, β, g²_ν, ν̂
end