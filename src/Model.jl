# Define a struct to encapsulate the ACB model parameters and state
mutable struct ACB_model
    y::Vector{Float64}
    X::Matrix{Float64}
    ω_ν::Float64
    α_ν::Float64
    β_ν::Float64
    g²_ν::Vector{Float64}
    μ::Vector{Float64}
    ω̄::Vector{Float64}
    β::Matrix{Float64}
    ξ::Vector{Float64}
    c::Vector{Float64}
    γ::Vector{Vector{Float64}}
    g²::Matrix{Float64}
    constant::Bool

    function ACB_model(y::Vector{Float64}, X::Matrix{Float64},ω_ν::Float64, α_ν::Float64, β_ν::Float64,  μ::Vector{Float64}, ω̄::Vector{Float64}, ξ::Vector{Float64}, c::Vector{Float64}, γ::Vector{Vector{Float64}},  g²::Matrix{Float64}; constant = false)

        β = zeros(size(X))
        g²_ν = zeros(size(y))
        new(y, X, ω_ν, α_ν, β_ν,g²_ν, μ, ω̄, β, ξ, c, γ,  g², constant)
    end

    function ACB_model(y::Vector{Float64}, X::Matrix{Float64},ω_ν::Float64, α_ν::Float64, β_ν::Float64,  ω̄::Vector{Float64}, ξ::Vector{Float64}, c::Vector{Float64}, γ::Vector{Vector{Float64}}; constant = true)
        
        μ = zeros(size(X, 2))
        g² = similar(X)

        for i in 1:size(X, 2)
            if i == 1 && constant                
                continue
            end
            result = GARCH_volatilities(X[:, i], GARCH{1, 1})

            g²[:,i] .= result[1] # g2_t+1 volatilities

            μ[i] = result[2]

        end
        ACB_model(y, X, ω_ν, α_ν, β_ν, μ, ω̄,  ξ, c, γ, g²; constant = constant)
    end
end

# Function to calculate y_t based on the current model state
function calculate_yt(model::ACB_model, x::Vector{Float64}, g_t::Float64, η_t::Float64)
    y_t = dot(model.β, x) + g_t * η_t
    return y_t
end

# Update β_i for each parameter
function update_βi(model::ACB_model, i::Int, νₜ::Float64, xᵢₜ::Float64, μₜ::Vector{Float64}, g²ₜ::Vector{Float64}, zₜ::Vector{Float64})
    βᵢₜ₊₁ = model.ω̄[i] + model.ξ[i] * (νₜ * xᵢₜ / (μₜ[i]^2 + g²ₜ[i])) +
            model.cᵢ[i] * βᵢₜ[i] + dot(model.γᵢ[i], zₜ)
    return βᵢₜ₊₁
end

# Update g_t2 for the next state
function update_g²(model::ACB_model, gₜ::Float64)
    g²ₜ₊₁ = model.ω + model.α * model.ν²ₜ + model.β * g²ₜ
    return g²ₜ₊₁
end

# Function to calculate x_i,t for each input
function calculate_xit(model::ACB_model, i::Int, gi_t::Float64, εi_t::Float64)
    xi_t = model.μ₀i[i] + gi_t * εi_t
    return xi_t
end

# Update g_i,t^2 for each input parameter
function update_g2i(model::ACB_model, i::Int, xi_t::Float64)
    g2i_t1 = model.ω₀i[i] + model.α₀i[i] * ((xi_t - model.μ₀i[i])^2) + model.β₀i[i] * model.g_i_t2[i]
    return g2i_t1
end

