module Settings 
    const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    import Pkg ; Pkg.activate(PROJECT_ROOT) ; Pkg.instantiate()
    using DotEnv ; DotEnv.load!() 

    include("Systems.jl"); include("forward_backward_sweep.jl")
    const name_to_func = Dict(
        "SE1" => Systems.SE1,
        "SE2" => Systems.SE2,
        "Modified SE1" => Systems.Modified_SE1,
        "Modified SE2" => Systems.Modified_SE2,
        "MidPoint" => Systems.MidPoint,
        "Modified MidPoint" => Systems.Modified_MidPoint,
    )

    const EDUCATED_GUESS_CHOICES = ["Random", "RK4ForwardBackward", "ForwardBackward"]
    const educated_guess = Dict(
        "RK4ForwardBackward" => forward_backward_sweep_module.RK4_forward_backward_sweep,
        "ForwardBackward" => forward_backward_sweep_module.forward_backward_sweep,
        "Random" => (u :: Vector{Float64}, a :: Float64, b :: Float64; y₀ :: Vector{Float64}, k_spring :: Float64 = 1.0, m :: Float64 = 1.0, x_d :: Vector{Float64}, N :: Int64 , α :: Float64, α₀ᴮᴮ :: Float64 = 1e-1, ϵ :: Float64 = 1e-6, α_max :: Float64 = 1., α_min :: Float64 = 1e-6, max_iter :: Int64 = -1, l₀ :: Float64) -> randn(Float64, 9N + 1)
    )


    const BATCH_SAVE_SIZE = parse(Int, get(ENV, "BATCH_SAVE_SIZE", "5"))
    const NTFY_TOPIC = get(ENV, "NTFY_TOPIC", nothing)
    const HTTP_URL = get(ENV, "HTTP_URL", nothing)

    const RESULTS_FOLDER = joinpath(PROJECT_ROOT, "Results")
    const COMPUTATION_RESULTS_FOLDER = joinpath(RESULTS_FOLDER, "Computation_Results/")
    const FIGURE_RESULTS_FOLDER = joinpath(RESULTS_FOLDER, "Figure_Results/")
    const SIMULATIONS_FOLDER = joinpath(RESULTS_FOLDER, "Simulations/")

    const NEWTON_MAX_ITER = parse(Int, get(ENV, "NEWTON_MAX_ITER", "150"))
    const NEWTON_TOL = parse(Float64, get(ENV, "NEWTON_TOL", "1e-10"))
    
    mkpath(COMPUTATION_RESULTS_FOLDER)
    mkpath(FIGURE_RESULTS_FOLDER)
    mkpath(SIMULATIONS_FOLDER)
end