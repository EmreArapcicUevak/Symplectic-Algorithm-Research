using Revise, BenchmarkTools, Metal
include("../Modules/NewtonMethodModule.jl")
include("../Modules/Systems.jl")

f(X :: Vector{Float32}) = @. X^2 - 2X
f_gpu(X :: Metal.MtlArray{Float32}) = @. X^2 - 2X
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120
@benchmark NewtonMethodModule.AproximateJacobian(f, ones(Float32,9999))
@benchmark NewtonMethodModule.AproximateJacobianCentral(f, ones(Float32,9999))

f(X :: Vector{Float32}) = @. X * X
f(X :: Metal.MtlArray{Float32}) =  @. X * X 

@benchmark NewtonMethodModule.MultiDimentionalNewtonMethod(f, x -> NewtonMethodModule.AproximateJacobian(f, x), ones(Float32, 400) .* -3)

@benchmark NewtonMethodModule.AproximateJacobian(f, Metal.MtlArray{Float32}(ones(Float32, 9001) .* -3))
@benchmark NewtonMethodModule.AproximateJacobian(f, ones(Float32, 9001) .* -3)



x₀_gpu = Metal.MtlArray(ones(Float32, 400) .* -3)

NewtonMethodModule.MultiDimentionalNewtonMethod(f, x -> NewtonMethodModule.AproximateJacobian(f, x), x₀_gpu)

n = 800
x₀ = ones(Float64, 9n + 1) .* -3
x₀_m = x₀[1:end-1]

@benchmark Systems.SE1(x₀; N = n, α = 10.0, m = 1., k = 1., a = Float64[0, 1], t₀ = 0.0, T = 10.0, l₀ = √2, x_d = Float64[0,2])
@benchmark Systems.SE2(x₀; N = n, α = 10.0, m = 1., k = 1., a = Float64[0, 1], t₀ = 0.0, T = 10.0, l₀ = √2, x_d = Float64[0,2])
@benchmark Systems.Modified_SE1(x₀_m; N = n, α = 10.0, m = 1., k = 1., a = Float64[0, 1], t₀ = 0.0, T = 10.0, l₀ = √2, x_d = Float64[0,2])
@benchmark Systems.Modified_SE2(x₀_m; N = n, α = 10.0, m = 1., k = 1., a = Float64[0, 1], t₀ = 0.0, T = 10.0, l₀ = √2, x_d = Float64[0,2])