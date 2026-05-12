module forward_backward_sweep_module

    using LinearAlgebra
    include("Systems.jl")

    export forward_backward_sweep
    function forward_backward_sweep(u :: Vector{Float64}, a :: Float64, b :: Float64; y₀ :: Vector{Float64}, k_spring :: Float64 = 1.0, m :: Float64 = 1.0, x_d :: Vector{Float64}, N :: Int64 , α :: Float64, α₀ᴮᴮ :: Float64 = 1e-1, ϵ :: Float64 = 1e-6, α_max :: Float64 = 1., α_min :: Float64 = 1e-6, max_iter :: Int64 = -1, l₀ :: Float64)
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
        #local l₀ = norm(x₀ - x_d)
        local h = (b - a)/N 
        local αᴮᴮ
        local sᵏ⁻¹, zᵏ⁻¹

        costs = Float64[]
        step_lenghts = Float64[]

        for k ∈ Iterators.countfrom(0)
            # Forward sweep
            for i ∈ 1:N
                y[i + 1] = y[i] + h * Systems.Hp(yₙ = y[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m)
            end

            # Backward Sweep 
            for i ∈ N:-1:1
                p[i] = p[i + 1] + h * Systems.Hy(yₙ = y[i + 1], pₙ = p[i + 1], uₙ = uᵏ[i + 1], k = k_spring, l₀ = l₀, m = m, x_d = x_d)
            end


            gᵏ = [Systems.Hu(yₙ = y[i], pₙ = p[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m, α = α) for i ∈ 1:N+1]

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

                αᴮᴮ = min(max(αᴮᴮ, α_min), αₜ)
            end
            push!(step_lenghts, αᴮᴮ)
            if length(step_lenghts) > 3 popfirst!(step_lenghts) end

            uᵏ⁺¹ = [uᵏ[i] - αᴮᴮ * gᵏ[i] for i ∈ 1:N+1]
            
            push!(costs, h*sum(Systems.Loss(yₙ = y[i], uₙ = uᵏ[i], α = α, x_d = x_d) for i ∈ 1:N))
            #println("$(costs[end]) - $(k)")

            u_update_norm =norm(uᵏ⁺¹ - uᵏ)
            g_norm = norm(gᵏ)


            if (u_update_norm < ϵ || g_norm < ϵ) || k == max_iter
                return vcat(
                    (t[3:4] for t in y[2:end])..., # x trajectory
                    (t[1:2] for t in y[2:end])..., # v trajectory
                    (t[1:2] for t in p[1:end-1])..., # λ trajectory
                    (t[3:4] for t in p[1:end-1])..., # μ trajectory
                    uᵏ
                ), costs, g_norm, step_lenghts
            end

            uᵏ⁻¹, gᵏ⁻¹ = uᵏ, gᵏ
            uᵏ = uᵏ⁺¹
        end
    end

    export RK4_forward_backward_sweep
    function RK4_forward_backward_sweep(u :: Vector{Float64}, a :: Float64, b :: Float64; y₀ :: Vector{Float64}, k_spring :: Float64 = 1.0, m :: Float64 = 1.0, x_d :: Vector{Float64}, N :: Int64 , α :: Float64, α₀ᴮᴮ :: Float64 = 1e-1, ϵ :: Float64 = 1e-6, α_max :: Float64 = 1., max_iter :: Int64 = -1, l₀ :: Float64)
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

        @views v₀, x₀  = y₀[1:2], y₀[3:4]
        #local l₀ = norm(x₀ - x_d)
        local h = (b - a)/N 
        local αᴮᴮ
        local sᵏ⁻¹, zᵏ⁻¹

        local costs = Float64[]
        local step_lenghts = Float64[]
        for k ∈ Iterators.countfrom(0)
            # Forward sweep
            for i ∈ 1:N
                local uₘ = (uᵏ[i] + uᵏ[i + 1]) / 2

                local k₁ = Systems.Hp(yₙ = y[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m)
                local k₂ = Systems.Hp(yₙ = y[i] + h/2 * k₁, uₙ = uₘ, k = k_spring, l₀ = l₀, m = m)
                local k₃ = Systems.Hp(yₙ = y[i] + h/2 * k₂, uₙ = uₘ, k = k_spring, l₀ = l₀, m = m)
                local k₄ = Systems.Hp(yₙ = y[i] + h * k₃, uₙ = uᵏ[i + 1], k = k_spring, l₀ = l₀, m = m)

                y[i + 1] = y[i] + h/6 * (k₁ + 2k₂ + 2k₃ + k₄)
            end

            # Backward Sweep 
            for i ∈ N:-1:1
                local yₘ, uₘ = (y[i + 1] + y[i]) / 2, (uᵏ[i + 1] + uᵏ[i]) / 2

                local l₁ = -Systems.Hy(yₙ = y[i + 1], uₙ = uᵏ[i + 1], pₙ = p[i + 1], k = k_spring, l₀ = l₀, m = m, x_d = x_d)
                local l₂ = -Systems.Hy(yₙ = yₘ, uₙ = uₘ, pₙ = p[i + 1] - h/2 * l₁, k = k_spring, l₀ = l₀, m = m, x_d = x_d)
                local l₃ = -Systems.Hy(yₙ = yₘ, uₙ = uₘ, pₙ = p[i + 1] - h/2 * l₂, k = k_spring, l₀ = l₀, m = m, x_d = x_d)
                local l₄ = -Systems.Hy(yₙ = y[i], uₙ = uᵏ[i], pₙ = p[i + 1] - h * l₃, k = k_spring, l₀ = l₀, m = m, x_d = x_d)

                p[i] = p[i + 1] - h/6 * (l₁ + 2l₂ + 2l₃ + l₄)
            end

            gᵏ = [Systems.Hu(yₙ = y[i], pₙ = p[i], uₙ = uᵏ[i], k = k_spring, l₀ = l₀, m = m, α = α) for i ∈ 1:N+1]

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

                #αᴮᴮ = min(max(αᴮᴮ, α_min), αₜ)
                αᴮᴮ = min(αᴮᴮ, αₜ)  
            end
            #println("αᴮᴮ = $(αᴮᴮ), ||gᵏ|| = $(norm(gᵏ))")
            push!(step_lenghts, αᴮᴮ)
            if length(step_lenghts) > 3 popfirst!(step_lenghts) end
            
            uᵏ⁺¹ = [uᵏ[i] - αᴮᴮ * gᵏ[i] for i ∈ 1:N+1]
            
            push!(costs, h*sum(Systems.Loss(yₙ = y[i], uₙ = uᵏ[i], α = α, x_d = x_d) for i ∈ 1:N))

            u_update_norm =norm(uᵏ⁺¹ - uᵏ)
            g_norm =norm(gᵏ)

            if (u_update_norm < ϵ || g_norm < ϵ) || k == max_iter
                return vcat(
                    (t[3:4] for t in y[2:end])..., # x trajectory
                    (t[1:2] for t in y[2:end])..., # v trajectory
                    (t[1:2] for t in p[1:end-1])..., # λ trajectory
                    (t[3:4] for t in p[1:end-1])..., # μ trajectory
                    uᵏ
                ), costs, g_norm, step_lenghts
            end

            uᵏ⁻¹, gᵏ⁻¹ = uᵏ, gᵏ
            uᵏ = uᵏ⁺¹
        end
    end

end