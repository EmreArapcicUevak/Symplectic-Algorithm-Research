using Test, Random, LinearAlgebra, BenchmarkTools

include("../Systems.jl")
include("../../Modules/Systems.jl")
include("../../Modules/NewtonMethodModule.jl")
include("../NewtonMethodModule.jl")


# Settings
N = 200
α = 1.0
m = 1.0
k = 10.0
a = Float64[0.0, -1.0]
t₀ = 0.0
T = 20.0
x₀ = Float64[1.0, 1.0]
x_d = Float64[2.0, 0.0]
l₀ = norm(x₀ - x_d)
####

params = Dict(
    :method => Systems_GPU.SE1,
    :N => N,
    :α => α,
    :m => m,
    :k => k,
    :a => a,
    :t₀ => t₀,
    :T => T,
    :x₀ => x₀,
    :l₀ => l₀,
    :x_d => x_d,
)

function CheckMethodAccuracy(MethodCPU::Function, MethodGPU::Function, Y_size=9 * N + 1)
    name = nameof(MethodCPU)
    println("STARTING TEST OF $name")
    @testset "$name Accuracy Gpu test" begin
        for i in 1:10
            Random.seed!(i)
            Y = rand(Float64, Y_size)
            R_julia = MethodCPU(Y; N=N, α=α, m=m, k=k, a=a, t₀=t₀, T=T, x₀=x₀, l₀=l₀, x_d=x_d)
            R_hip = MethodGPU(Y; N=N, α=α, m=m, k=k, a=a, t₀=t₀, T=T, x₀=x₀, l₀=l₀, x_d=x_d)

            max_diff = maximum(abs.(R_julia - R_hip))
            norm_diff = norm(R_julia - R_hip)
            println("Seed : $i")
            println("Norm diff : $norm_diff")
            println("Max diff : $max_diff")
            @test norm_diff < 1e-10
        end
    end
end

function BenchmarkMethod(MethodCPU::Function, MethodGPU::Function, Y_size::Int, seed=1)
    Random.seed!(seed)
    name = nameof(MethodCPU)
    println("BENCHMARKING $name")
    Y = rand(Float64, Y_size)
    println("Julia CPU time :")
    @btime R_julia = $MethodCPU($Y; N=$N, α=$α, m=$m, k=$k, a=$a, t₀=$t₀, T=$T, x₀=$x₀, l₀=$l₀, x_d=$x_d)
    println("HIP GPU time :")
    @btime R_hip = $MethodGPU($Y; N=$N, α=$α, m=$m, k=$k, a=$a, t₀=$t₀, T=$T, x₀=$x₀, l₀=$l₀, x_d=$x_d)
end

function CheckJacobianAccuracy(JacobianCPU::Function, MethodCPU::Function, JacobianGPU::Function, MethodGPU::Function, Y_size=9 * N + 1)
    println("$(nameof(JacobianCPU))  $(nameof(MethodCPU))  $(nameof(JacobianGPU))  $(nameof(MethodGPU)) Accuracy Test")
    @testset "Accuracy Test Result" begin
        for i in 1:10

            Random.seed!(i)
            Y = rand(Float64, Y_size)

            F_cpu = x -> MethodCPU(x; N=N, α=α, m=m, k=k, a=a, t₀=t₀, T=T, x₀=x₀, l₀=l₀, x_d=x_d)
            J_cpu = JacobianCPU(F_cpu, Y)

            params[:method] = MethodGPU
            J_gpu = JacobianGPU(params, Y)

            max_diff, idx = findmax(abs.(J_cpu - J_gpu))
            norm_diff = norm(J_cpu - J_gpu)
            println("seed=$i  max_diff=$max_diff  at $idx")
            println("norm_diff = $norm_diff")
            println("  J_cpu=$(J_cpu[idx])  J_gpu=$(J_gpu[idx])")
            @test norm_diff < 1e-7
        end
    end
end

function BenchmarkJacobian(JacobianCPU::Function, MethodCPU::Function, JacobianGPU::Function, MethodGPU::Function, Y_size=9 * N + 1, seed=1)
    Random.seed!(seed)
    println("$(nameof(JacobianCPU))  $(nameof(MethodCPU))  $(nameof(JacobianGPU))  $(nameof(MethodGPU)) Benchmark")


    Y = rand(Float64, Y_size)
    println("Julia CPU time :")
    F_cpu = x -> MethodCPU(x; N=N, α=α, m=m, k=k, a=a, t₀=t₀, T=T, x₀=x₀, l₀=l₀, x_d=x_d)
    @btime J_cpu = $JacobianCPU($F_cpu, $Y)
    println("HIP GPU time :")
    params[:method] = MethodGPU
    @btime J_gpu = $JacobianGPU($params, $Y)
end

if false
    CheckMethodAccuracy(Systems.SE1, Systems_GPU.SE1)
    CheckMethodAccuracy(Systems.SE2, Systems_GPU.SE2)
    CheckMethodAccuracy(Systems.Modified_SE1, Systems_GPU.Modified_SE1)
    CheckMethodAccuracy(Systems.Modified_SE2, Systems_GPU.Modified_SE2)
    CheckMethodAccuracy(Systems.MidPoint, Systems_GPU.MidPoint)
    CheckMethodAccuracy(Systems.Modified_MidPoint, Systems_GPU.Modified_MidPoint)
end


if false
    BenchmarkMethod(Systems.SE1, Systems_GPU.SE1, 9 * N + 1)
    BenchmarkMethod(Systems.SE2, Systems_GPU.SE2, 9 * N + 1)
    BenchmarkMethod(Systems.MidPoint, Systems_GPU.MidPoint, 9 * N + 1)
    BenchmarkMethod(Systems.Modified_SE1, Systems_GPU.Modified_SE1, 9 * N)
    BenchmarkMethod(Systems.Modified_SE2, Systems_GPU.Modified_SE2, 9 * N)
    BenchmarkMethod(Systems.Modified_MidPoint, Systems_GPU.Modified_MidPoint, 9 * N)
end



#CheckJacobianAccuracy(NewtonMethodModule.AproximateJacobianCentral, Systems.SE2, NewtonMethodModule_GPU.AproximateJacobianCentral, Systems_GPU.SE2)
BenchmarkJacobian(NewtonMethodModule.AproximateJacobianCentral, Systems.SE2, NewtonMethodModule_GPU.AproximateJacobianCentral, Systems_GPU.SE2)

