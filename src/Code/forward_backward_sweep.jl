using Pkg; Pkg.instantiate()

include("../Modules/NewtonMethodModule.jl"); include("../Modules/Systems.jl"); include("../Modules/CLI_Param.jl")
using MAT, Term.Progress, Serialization, Base.Threads, BenchmarkTools, Serialization, LinearAlgebra, CSV, DataFrames
using Plots


function H(; yₙ :: Vector{Float64}, pₙ :: Vector{Float64}, uₙ :: Float64, k :: Float64, l₀ :: Float64, m :: Float64, α :: Float64, x_d :: Vector{Float64} , a :: Vector{Float64} = Float64[0, -1])
    @views xₙ, vₙ = yₙ[1:2], yₙ[3:4]
    @views λₙ, μₙ = pₙ[1:2], pₙ[3:4]

    local u_to_x = xₙ - Float64[uₙ, 0.0]
    local L = norm(u_to_x) 

    local f = (norm(xₙ - x_d)^2 + α*uₙ^2) / 2
    local δUδx = k * (L - l₀) / L * u_to_x - a

    return f + dot(λₙ, (-1/m) * δUδx) + dot(μₙ, vₙ)
end

function Hp(; yₙ :: Vector{Float64}, uₙ :: Float64, k :: Float64, l₀ :: Float64, m :: Float64, a ::Vector{Float64} = Float64[0,-1])
    @views xₙ, vₙ = yₙ[1:2], yₙ[3:4]

    local u_to_x = xₙ - Float64[uₙ, 0.0]
    local L = norm(u_to_x)

    result = similar(yₙ, 4)
    result[1:2] .= vₙ
    result[3:4] .= 1/m * a - k / m * ( L - l₀ ) / L * (u_to_x)

    return result
end

function Hy(; yₙ :: Vector{Float64}, pₙ :: Vector{Float64}, uₙ :: Float64, k :: Float64, l₀ :: Float64, m :: Float64, x_d :: Vector{Float64})
    @views xₙ, vₙ = yₙ[1:2], yₙ[3:4]
    @views λₙ, μₙ = pₙ[1:2], pₙ[3:4]
    local u_to_x = xₙ - Float64[uₙ, 0.0]
    local L = norm(u_to_x) 

    result = similar(pₙ, 4)
    result[1:2] .= μₙ
    result[3:4] .= (xₙ - x_d) - (k * l₀ / (m * L^3) * u_to_x * u_to_x' + k/m * (L - l₀)/L * Matrix{Float64}(I, 2, 2)) * λₙ

    return result
end

function Hu(; yₙ :: Vector{Float64}, pₙ :: Vector{Float64}, uₙ :: Float64, k :: Float64, l₀ :: Float64, m :: Float64, α :: Float64)
    @views xₙ, vₙ = yₙ[1:2], yₙ[3:4]
    @views λₙ, μₙ = pₙ[1:2], pₙ[3:4]
    local u_to_x = xₙ - Float64[uₙ, 0.0]
    local L = norm(u_to_x) 

    local c1 = -k * l₀ / (m * L^3)
    local c2 = - k / m * (L - l₀) / L 
    local e1 = Float64[1,0]

    return α * uₙ - dot(λₙ, (c1 * dot(e1, u_to_x)) * u_to_x + c2 * e1)
end


function Loss(; yₙ :: Vector{Float64}, uₙ :: Float64, α :: Float64, x_d :: Vector{Float64})
    @views xₙ, vₙ = yₙ[1:2], yₙ[3:4]
    return (norm(xₙ - x_d, 2)^2 + α * uₙ ^ 2) / 2
end

function forward_backward_sweep(u :: Vector{Float64}, a :: Float64, b :: Float64; y₀ :: Vector{Float64}, k_spring :: Float64 = 1.0, m :: Float64 = 1.0, x_d :: Vector{Float64}, N :: Int64 , α :: Float64, α₀ᴮᴮ :: Float64 = 1e-1, ϵ :: Float64 = 1e-6, α_max :: Float64 = 1., max_iter :: Int64 = -1)
    @assert 0.0 < α_max
    @assert ϵ > 0.0
    @assert α > 0.0
    @assert α₀ᴮᴮ > 0.0
    @assert N > 0
    @assert length(u) == N+1
    @assert a < b
    @assert α > 0.0

    local y = Vector{Vector{Float64}}(undef, N + 1); y[1] = y₀
    local p = Vector{Vector{Float64}}(undef, N + 1); p[end] = zeros(Float64, 4)
    local uᵏ = copy(u)
    local gᵏ, gᵏ⁻¹,uᵏ⁻¹

    @views x₀, v₀ = y₀[1:2], y₀[3:4]
    local l₀ = norm(x₀ - x_d)
    local h = (b - a)/N 
    local αᴮᴮ
    local sᵏ⁻¹, zᵏ⁻¹

    costs = Float64[]

    for k ∈ Iterators.countfrom(0)
        # Forward sweep
        for i ∈ 1:N
            y[i + 1] = y[i] + h * Hp(yₙ = y[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m)
        end

        # Backward Sweep 
        for i ∈ N:-1:1
            p[i] = p[i + 1] + h * Hy(yₙ = y[i + 1], pₙ = p[i + 1], uₙ = uᵏ[i + 1], k = k_spring, l₀ = l₀, m = m, x_d = x_d)
        end


        gᵏ = [Hu(yₙ = y[i], pₙ = p[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m, α = α) for i ∈ 1:N+1]

        if k == 0
            αᴮᴮ =  α₀ᴮᴮ
        else
            sᵏ⁻¹ = uᵏ - uᵏ⁻¹
            zᵏ⁻¹ = gᵏ - gᵏ⁻¹
            αₜ = α_max / norm(gᵏ)

            #αᴮᴮ =  dot(sᵏ⁻¹, sᵏ⁻¹) / dot(sᵏ⁻¹, zᵏ⁻¹)
            αᴮᴮ =  - dot(sᵏ⁻¹, zᵏ⁻¹) / norm(zᵏ⁻¹)^2
            if αᴮᴮ <= 0
                αᴮᴮ =  norm(sᵏ⁻¹) / norm(zᵏ⁻¹)
            end

            αᴮᴮ = min(αᴮᴮ, αₜ)
        end
        
        uᵏ⁺¹ = [uᵏ[i] - αᴮᴮ * gᵏ[i] for i ∈ 1:N+1]
        
        push!(costs, h*sum(Loss(yₙ = y[i], uₙ = uᵏ⁺¹[i], α = α, x_d = x_d) for i ∈ 1:N))
        #println("$(costs[end]) - $(k)")

        #u_update_norm =norm(uᵏ⁺¹ - uᵏ)
        #g_norm =norm(gᵏ)
        u_update_norm = maximum(norm.(uᵏ⁺¹ - uᵏ))
        g_norm = maximum(norm.(gᵏ))

        if u_update_norm < ϵ || g_norm < ϵ || k == max_iter
            #return uᵏ⁺¹, costs
            return vcat(
                (t[1:2] for t in y[2:end])...,
                (t[3:4] for t in y[2:end])...,
                (t[1:2] for t in p[1:end-1])...,
                (t[3:4] for t in p[1:end-1])...,
                uᵏ⁺¹
            ), costs
            #return vcat(y[2:end]...,p[1:end-1]...,uᵏ⁺¹), costs
        end

        uᵏ⁻¹, gᵏ⁻¹ = uᵏ, gᵏ
        uᵏ = uᵏ⁺¹
    end
end

function get_initial_guess(N :: Integer, x₀ :: Vector{Float64}, x_d :: Vector{Float64}, m :: Float64, k :: Float64, l₀ :: Float64) :: Vector{Float64}
  @assert length(x₀) == 2
  # Compute educated starting spring pos
  local temp_f = u -> -k * (norm(x₀ - [u, 0.0]) - l₀) * (x₀[2]) / norm(x₀ - [u, 0.0]) - m
  local educated_u
  try
    if x₀[1] >= x_d[1]
      educated_u = NewtonMethodModule.QuasiNewtonMethod(temp_f, x₀[1], x₀[1] + 0.5 ; maxIterations = 100, δ = 1e-10, ϵ = 1e-10).c
    else
      educated_u = NewtonMethodModule.QuasiNewtonMethod(temp_f, x₀[1], x₀[1] - 0.5 ; maxIterations = 100, δ = 1e-10, ϵ = 1e-10).c
    end
  catch e 
    #@warn "Educated guess computation failed, using 0.0 as guess" x0=x₀
    educated_u = x₀[1]
  end
  local x₀_guess = rand(Float64, 9 * N + 1) * 2
  local stable_height = l₀ - m / k
  local stable_final_pos = Float64[0.0, stable_height]


  x₀_guess[1:N] .= repeat(x₀, N ÷ 2) # Change x₀ till xₙ/2
  x₀_guess[N+1:2N] .= repeat(stable_final_pos, N ÷ 2)  # Change xₙ/2 till xₙ
  x₀_guess[2N+1:2:4N] .= 0.0  # Change x component of v to 0
  x₀_guess[8N+1:end] .= 0.0  # Change u to 0
  x₀_guess[8N+1:(17N + 3) ÷ 2] .= LinRange(educated_u, 0.0, (N+3) ÷ 2)  # Change half of the u's to educated guess

  return x₀_guess
end


function RK4_forward_backward_sweep(u :: Vector{Float64}, a :: Float64, b :: Float64; y₀ :: Vector{Float64}, k_spring :: Float64 = 1.0, m :: Float64 = 1.0, x_d :: Vector{Float64}, N :: Int64 , α :: Float64, α₀ᴮᴮ :: Float64 = 1e-1, ϵ :: Float64 = 1e-6, α_max :: Float64 = 1., max_iter :: Int64 = -1)
    @assert 0.0 < α_max
    @assert ϵ > 0.0
    @assert α > 0.0
    @assert α₀ᴮᴮ > 0.0
    @assert N > 0
    @assert length(u) == N+1
    @assert a < b
    @assert α > 0.0

    local y = Vector{Vector{Float64}}(undef, N + 1); y[1] = y₀
    local p = Vector{Vector{Float64}}(undef, N + 1); p[end] = zeros(Float64, 4)
    local uᵏ = copy(u)
    local gᵏ, gᵏ⁻¹,uᵏ⁻¹

    @views x₀, v₀ = y₀[1:2], y₀[3:4]
    local l₀ = norm(x₀ - x_d)
    local h = (b - a)/N 
    local αᴮᴮ
    local sᵏ⁻¹, zᵏ⁻¹

    costs = Float64[]

    for k ∈ Iterators.countfrom(0)
        # Forward sweep
        for i ∈ 1:N
            local uₘ = (uᵏ[i] + uᵏ[i + 1]) / 2

            local k₁ = Hp(yₙ = y[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m)
            local k₂ = Hp(yₙ = y[i] + h/2 * k₁, uₙ = uₘ, k = k_spring, l₀ = l₀, m = m)
            local k₃ = Hp(yₙ = y[i] + h/2 * k₂, uₙ = uₘ, k = k_spring, l₀ = l₀, m = m)
            local k₄ = Hp(yₙ = y[i] + h * k₃, uₙ = uᵏ[i + 1], k = k_spring, l₀ = l₀, m = m)

            y[i + 1] = y[i] + h/6 * (k₁ + 2k₂ + 2k₃ + k₄)
        end

        # Backward Sweep 
        for i ∈ N:-1:1
            local yₘ, uₘ = (y[i + 1] + y[i]) / 2, (uᵏ[i + 1] + uᵏ[i]) / 2

            local l₁ = -Hy(yₙ = y[i + 1], uₙ = uᵏ[i + 1], pₙ = p[i + 1], k = k_spring, l₀ = l₀, m = m, x_d = x_d)
            local l₂ = -Hy(yₙ = yₘ, uₙ = uₘ, pₙ = p[i + 1] - h/2 * l₁, k = k_spring, l₀ = l₀, m = m, x_d = x_d)
            local l₃ = -Hy(yₙ = yₘ, uₙ = uₘ, pₙ = p[i + 1] - h/2 * l₂, k = k_spring, l₀ = l₀, m = m, x_d = x_d)
            local l₄ = -Hy(yₙ = y[i], uₙ = uᵏ[i], pₙ = p[i + 1] - h * l₃, k = k_spring, l₀ = l₀, m = m, x_d = x_d)

            p[i] = p[i + 1] - h/6 * (l₁ + 2l₂ + 2l₃ + l₄)
        end


        gᵏ = [Hu(yₙ = y[i], pₙ = p[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m, α = α) for i ∈ 1:N+1]

        if k == 0
            αᴮᴮ =  α₀ᴮᴮ
        else
            sᵏ⁻¹ = uᵏ - uᵏ⁻¹
            zᵏ⁻¹ = gᵏ - gᵏ⁻¹
            αₜ = α_max / norm(gᵏ)

            #αᴮᴮ =  dot(sᵏ⁻¹, sᵏ⁻¹) / dot(sᵏ⁻¹, zᵏ⁻¹)
            αᴮᴮ =  - dot(sᵏ⁻¹, zᵏ⁻¹) / norm(zᵏ⁻¹)^2
            if αᴮᴮ <= 0
                αᴮᴮ =  norm(sᵏ⁻¹) / norm(zᵏ⁻¹)
            end

            αᴮᴮ = min(αᴮᴮ, αₜ)
        end
        
        uᵏ⁺¹ = [uᵏ[i] - αᴮᴮ * gᵏ[i] for i ∈ 1:N+1]
        
        push!(costs, h*sum(Loss(yₙ = y[i], uₙ = uᵏ⁺¹[i], α = α, x_d = x_d) for i ∈ 1:N))
        #println("$(costs[end]) - $(k)")

        u_update_norm =norm(uᵏ⁺¹ - uᵏ)
        g_norm =norm(gᵏ)
        #u_update_norm = maximum(norm.(uᵏ⁺¹ - uᵏ))
        #g_norm = maximum(norm.(gᵏ))

        if u_update_norm < ϵ || g_norm < ϵ || k == max_iter
            #return uᵏ⁺¹, costs
            return vcat(
                (t[1:2] for t in y[2:end])...,
                (t[3:4] for t in y[2:end])...,
                (t[1:2] for t in p[1:end-1])...,
                (t[3:4] for t in p[1:end-1])...,
                uᵏ⁺¹
            ), costs
            #return vcat(y[2:end]...,p[1:end-1]...,uᵏ⁺¹), costs
        end

        uᵏ⁻¹, gᵏ⁻¹ = uᵏ, gᵏ
        uᵏ = uᵏ⁺¹
    end
end


N = 200
x₀ = Float64[1, 4]
x_d = Float64[0, 10]
m = 1.
k = 0.5
l₀ = norm(x₀ - x_d)
α = 10.


initial_guess = get_initial_guess(N, x₀, x_d, m, k, l₀)
u_guess = [Systems.u(initial_guess, i, N) for i ∈ 0:N]

fb_res, costs = forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = vcat(x₀, Float64[0, 0]), x_d = x_d, N = N, m = m, k_spring = k, ϵ = 1e-6, max_iter = 5000)
u_fb = [Systems.u(fb_res, i, N) for i ∈ 0:N]
#u_fb = fb_res
costs


rk4_fb_res, costs_RK4 = RK4_forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = vcat(x₀, Float64[0, 0]), x_d = x_d, N = N, m = m, k_spring = k, ϵ = 1e-6, max_iter = 5000)
u_rk_fb = [Systems.u(rk4_fb_res, i, N) for i ∈ 0:N]

costs_RK4

plot(costs_RK4, lw = 3, label="RK4 forward backward")
plot!(costs, lw=3, label = "forward backward")
ylabel!("Residual Cost")
xlabel!("Iteration")

method_SE1 = x -> Systems.SE1(x; N = N, α = α ,m = m ,k = k, a = Float64[0,-1], t₀ = 0.0, T = 10.0, l₀ = l₀, x₀ = x₀, x_d = x_d)
results_SE1 = NewtonMethodModule.MultiDimentionalNewtonMethod(method_SE1, x -> NewtonMethodModule.AproximateJacobian(method_SE1, x), rk4_fb_res[1:end]; maxIterations = 500, δ = 0.5e-10, ϵ = 0.5e-12)
u_newton_SE1 = [Systems.u(results_SE1.c, i, N) for i ∈ 0:N-1]


method_MSE1 = x -> Systems.Modified_SE1(x; N = N, α = α ,m = m ,k = k, a = Float64[0,-1], t₀ = 0.0, T = 10.0, l₀ = l₀, x₀ = x₀, x_d = x_d)
results_MSE1 = NewtonMethodModule.MultiDimentionalNewtonMethod(method_MSE1, x -> NewtonMethodModule.AproximateJacobian(method_MSE1, x), rk4_fb_res[1:end-1]; maxIterations = 500, δ = 0.5e-10, ϵ = 0.5e-12)
u_newton_MSE1 = [Systems.u(results_MSE1.c, i, N) for i ∈ 0:N-1]


plot(u_fb, label="gradient method", lw=3)
plot!(u_rk_fb, label="RK4 gradient method", lw=3)
plot!(u_newton_SE1, label="newthon method SE1", lw=3)
plot!(u_newton_SE1, label="newthon method M.SE1", lw=3)
ylabel!("uₜ")
xlabel!("t")


H_fb = [
    H(
        yₙ = vcat(Systems.x(fb_res, i, N, x₀), Systems.v(fb_res, i, N)),
        pₙ = vcat(Systems.λ(fb_res, i, N), Systems.μ(fb_res, i, N)),
        uₙ = Systems.u(fb_res, i, N),
        k = k,
        l₀ = l₀,
        m = m,
        α = α,
        x_d = x_d
    ) 
    for i ∈ 0:N
]

H_rk4_fb = [
    H(
        yₙ = vcat(Systems.x(rk4_fb_res, i, N, x₀), Systems.v(rk4_fb_res, i, N)),
        pₙ = vcat(Systems.λ(rk4_fb_res, i, N), Systems.μ(rk4_fb_res, i, N)),
        uₙ = Systems.u(rk4_fb_res, i, N),
        k = k,
        l₀ = l₀,
        m = m,
        α = α,
        x_d = x_d
    ) 
    for i ∈ 0:N
]

H_MSE1 = [
    H(
        yₙ = vcat(Systems.x(results_MSE1.c, i, N, x₀), Systems.v(results_MSE1.c, i, N)),
        pₙ = vcat(Systems.λ(results_MSE1.c, i, N), Systems.μ(results_MSE1.c, i, N)),
        uₙ = Systems.u(results_MSE1.c, i, N),
        k = k,
        l₀ = l₀,
        m = m,
        α = α,
        x_d = x_d
    ) 
    for i ∈ 0:N-1
]

H_SE1 = [
    H(
        yₙ = vcat(Systems.x(results_SE1.c, i, N, x₀), Systems.v(results_SE1.c, i, N)),
        pₙ = vcat(Systems.λ(results_SE1.c, i, N), Systems.μ(results_SE1.c, i, N)),
        uₙ = Systems.u(results_SE1.c, i, N),
        k = k,
        l₀ = l₀,
        m = m,
        α = α,
        x_d = x_d
    ) 
    for i ∈ 0:N
]


plot(H_fb, lw = 3, label="forward backward", xlabel="t", ylabel = "Hₜ")
plot!(H_rk4_fb, lw = 3, label= "RK4 Forward Backward")
plot!(H_MSE1, lw = 3, label = "Modified SE1")
plot!(H_SE1, lw = 3, label = "SE1")

x_fb = [Systems.x(fb_res, i, N, x₀) for i ∈ 1:N]
x_rk4_fb = [Systems.x(rk4_fb_res, i, N, x₀) for i ∈ 1:N]
x_MSE1 = [Systems.x(results.c, i, N, x₀) for i ∈ 1:N]

x_fb .- x_rk4_fb

plot(norm.(x_fb .- x_MSE1), lw=3, label="forward backward vs Modified SE1")
plot!(norm.(x_rk4_fb .- x_MSE1), lw=3, label="RK4 forward backward vs Modified SE1")
plot!(norm.(x_rk4_fb .- x_fb), lw=3, label="RK4 forward backward vs forward backward")

v_fb = [Systems.v(fb_res, i, N) for i ∈ 1:N]
v_rk4_fb = [Systems.v(rk4_fb_res, i, N) for i ∈ 1:N]
v_MSE1 = [Systems.v(results.c, i, N) for i ∈ 1:N]

v_fb .- v_rk4_fb

plot(norm.(v_fb .- v_MSE1), lw=3, label="forward backward vs Modified SE1")
plot!(norm.(v_rk4_fb .- v_MSE1), lw=3, label="RK4 forward backward vs Modified SE1")
plot!(norm.(v_rk4_fb .- v_fb), lw=3, label="RK4 forward backward vs forward backward")

λ_fb = [Systems.λ(fb_res, i, N) for i ∈ 1:N]
λ_rk4_fb = [Systems.λ(rk4_fb_res, i, N) for i ∈ 1:N]
λ_MSE1 = [Systems.λ(results.c, i, N) for i ∈ 1:N]

plot(norm.(λ_fb .- λ_MSE1), lw=3, label="forward backward vs Modified SE1")
plot!(norm.(λ_rk4_fb .- λ_MSE1), lw=3, label="RK4 forward backward vs Modified SE1")
plot!(norm.(λ_rk4_fb .- λ_fb), lw=3, label="RK4 forward backward vs forward backward")


μ_fb = [Systems.μ(fb_res, i, N) for i ∈ 1:N]
μ_rk4_fb = [Systems.μ(rk4_fb_res, i, N) for i ∈ 1:N]
μ_MSE1 = [Systems.μ(results.c, i, N) for i ∈ 1:N]

plot(norm.(μ_fb .- μ_MSE1), lw=3, label="forward backward vs Modified SE1")
plot!(norm.(μ_rk4_fb .- μ_MSE1), lw=3, label="RK4 forward backward vs Modified SE1")
plot!(norm.(μ_rk4_fb .- μ_fb), lw=3, label="RK4 forward backward vs forward backward")

