module NewtonMethodModule
        using LinearAlgebra, Base.Threads, Metal
        export NewtonMethod, QuasiNewtonMethod, MultiDimentionalNewtonMethod, AproximateJacobian, AproximateJacobianCentral
    
        function NewtonMethod(f :: Function, f_prime :: Function, x₀ :: Real; δ = 1e-15, ϵ = 1e-15, maxIterations = 1000)
            for i ∈ 1:maxIterations
                local f_prime_x₀, f_x₀ = f_prime(x₀), f(x₀)

                local h = f_x₀ / f_prime_x₀
                if abs(h / x₀) ≤ ϵ || abs(f_x₀) ≤ δ
                    return (c = x₀,iterations = i)
                end

                x₀ -= h 
            end

            return nothing
        end

        function QuasiNewtonMethod(f :: Function, x₀ :: Real, x₁ :: Real; δ = 1e-15, ϵ = 1e-15, maxIterations = 1000)
            local f₀, f₁ = f(x₀), f(x₁)
            
            for i ∈ 1:maxIterations
                local xₖ = x₁ - (x₁ - x₀) * f₁ / (f₁ - f₀)
                x₀, x₁ = x₁, xₖ
                f₀, f₁ = f(x₀), f(x₁)

                println("k: $i, xₖ: $xₖ, f(xₖ): $(f(xₖ))")
                if abs(x₁ - x₀) ≤ ϵ || abs(f(x₁)) ≤ δ
                    return (c = x₁,iterations = i)
                end
            end

            return nothing
            
        end

        function MultiDimentionalNewtonMethod(F :: Function, J :: Function, x₀ :: Vector{T}; δ = 1e-10, ϵ = 1e-10, maxIterations = 1000, history::Union{Vector{Float64}, Nothing} = nothing, iteration_points :: Union{Vector{Vector{T}}, Nothing} = nothing) where T <: Real
            for i ∈ 1:maxIterations
                local J_x₀, F_x₀ = J(x₀), F(x₀)
                local Δ = - J_x₀ \ F_x₀
                local x₁ = x₀ + Δ

                Δ_norm = LinearAlgebra.norm(Δ, 2)
                F_x₁_norm = LinearAlgebra.norm(F(x₁), 2)
                #println("Iteration $i done out of $maxIterations")
                #println("$(Δ_norm), $(F_x₁_norm)")
                if history !== nothing push!(history, F_x₁_norm) end
                if iteration_points !== nothing push!(iteration_points, x₁) end
                if Δ_norm ≤ ϵ && F_x₁_norm ≤ δ return (c = x₁,iterations = i) end
                x₀ = x₁
            end

            return nothing
        end


        function MultiDimentionalNewtonMethod(F :: Function, J :: Function, x₀_gpu :: Metal.MtlArray{Float32}; δ = 1e-10, ϵ = 1e-10, maxIterations = 1000, history::Union{Vector{Float64}, Nothing} = nothing, iteration_points :: Union{Vector{Vector{T}}, Nothing} = nothing) where T <: Real
            # dimensions
            n = length(x₀_gpu)

            # Preallocate buffers on GPU
            J_gpu = Metal.MtlArray(zeros(Float32, n, n))
            F_gpu = Metal.MtlArray(zeros(Float32, n))
            Δ_gpu = Metal.MtlArray(zeros(Float32, n))
            x₁_gpu = Metal.MtlArray(zeros(Float32, n))

            for i ∈ 1:maxIterations
                 J_gpu .= J(x₀_gpu)
                @. F_gpu = F(x₀_gpu)
                @. Δ_gpu = - J_gpu \ F_gpu
                @. x₁_gpu = x₀_gpu + Δ_gpu

                Δ_norm_gpu = Float64(Metal.LinearAlgebra.norm(Δ, 2))
                F_x₁_norm_gpu = Float64(Metal.LinearAlgebra.norm(F(x₁_gpu), 2))

                if history !== nothing push!(history, F_x₁_norm) end
                if iteration_points !== nothing push!(iteration_points, x₁) end
                if Δ_norm ≤ ϵ && F_x₁_norm ≤ δ return (c = x₁,iterations = i) end

                x₀_gpu .= x₁_gpu
            end

            return nothing
        end

        function AproximateJacobian(F :: Function, x₀ :: Vector{Float64}; t = 1e-6 :: Float64) 
            Fx₀ = F(x₀)
            local J = zeros(T, length(Fx₀), length(x₀))

            Threads.@threads for i ∈ 1:length(x₀)
                local x = copy(x₀)
                x[i] += t

                @inbounds  J[:,i] .= (F(x) .- Fx₀) ./ t
            end

            return J
        end

        function AproximateJacobian(F :: Function, x₀ :: MtlArray{Float32}; t = 1f-6 :: Float32)
            Fx₀_gpu = F(x₀)
            # dimensions
            n = length(x₀)
            m = length(Fx₀_gpu)
            
            J_gpu = Metal.MtlMatrix{Float32}(undef, (m, n))
            basis_gpu = Metal.mtl(Matrix{Float32}(I, n, n)) 
            x_temp_gpu = similar(x₀)
            F_temp_gpu = Metal.MtlArray{Float32}(undef, m)
            
            for i ∈ 1:length(x₀)
                copyto!(x_temp_gpu, x₀)
                x_temp_gpu .+= t .* basis_gpu[:, i]   
                F_temp_gpu .= F(x_temp_gpu)

                @inbounds  J_gpu[:,i] .= (F_temp_gpu .- Fx₀_gpu) ./ t
            end

            return J_gpu
        end

        function AproximateJacobianCentral(F :: Function, x₀ :: Vector{T}; t = 1e-6 :: Float64) where T <: Real
            local N = length(x₀)
            local J = zeros(T, N, N)

            Fx₀ = F(x₀)
            Threads.@threads for i ∈ 1:length(x₀)
                local x_plus = copy(x₀)
                local x_minus = copy(x₀)
                x_plus[i] += t
                x_minus[i] -= t
                @inbounds J[:,i] = (F(x_plus) .- F(x_minus)) ./ (2 * t)
            end

            return J
        end
end