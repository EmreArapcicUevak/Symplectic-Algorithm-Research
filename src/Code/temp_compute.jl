using Pkg; Pkg.instantiate()

include("../Modules/NewtonMethodModule.jl"); include("../Modules/Systems.jl"); include("../Modules/CLI_Param.jl"); include("../Modules/forward_backward_sweep.jl"); include("../Modules/CLI_Param.jl"); include("../Modules/Settings.jl"); include("../Modules/Remote_Status_Notifier.jl")
using MAT, Term.Progress, Serialization, Base.Threads, BenchmarkTools, LinearAlgebra, CSV, DataFrames
using Plots, ArgParse, REPL.TerminalMenus

param_grid, output_file_name = CLI_Param.get_parameters(sort!(collect(keys(Settings.name_to_func))))
delete!(param_grid, :method)
delete!(param_grid, :educated_guess)

println(param_grid)

result_lock = ReentrantLock()

keys_ = collect(keys(param_grid))
values_ = [param_grid[key] for key in keys_]
lens = map(length, values_)
n_combinations = reduce(*, lens)
space = CartesianIndices(Tuple(lens))

result_folder = "Results/"
figure_results_folder = joinpath(result_folder, "figure_results/")

mkpath(figure_results_folder)

pbar = ProgressBar(); comp_job = addjob!(pbar,N = n_combinations, description = "Total Progress")
start!(pbar); render(pbar)

rows = Vector{Dict{Symbol, Any}}()
Threads.@threads for i ∈ 1:n_combinations
    local tup = ntuple(j -> values_[j][space[i][j]], length(values_))
    local paramaters = NamedTuple{Tuple(keys_)}(tup)

    local N = paramaters[:N]
    local x₀ = paramaters[:x₀]
    local x_d = paramaters[:x_d]
    local m = paramaters[:m]
    local k = paramaters[:k]
    local l₀ = x_d[2] - m / k
    local α = paramaters[:α]

    local file_name_base = "N=$N,x₀=$(x₀),x_d=$(x_d),m=$(m),k=$(k),α=$α,l₀=$(l₀)"

    local initial_guess = Systems.get_initial_guess(N, x₀, x_d, m, k, l₀)
    local u_guess = [Systems.u(initial_guess, i, N) for i ∈ 0:N]

    local fb_out = @timed forward_backward_sweep_module.forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = vcat(x₀, Float64[0, 0]), x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-14, max_iter = 500000)
    local fb_time = fb_out[:time]
    local fb_res, fb_cost = fb_out[:value]

    local rk4_out = @timed forward_backward_sweep_module.RK4_forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = vcat(x₀, Float64[0, 0]), x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-14, max_iter = 500000)
    local rk4_time = rk4_out[:time]
    local rk4_res, rk4_cost = rk4_out[:value]

    local p = plot(rk4_cost, lw = 3, label="RK4 forward backward") ; plot!(fb_cost, lw=3, label = "forward backward") ; ylabel!("Residual Cost") ; xlabel!("Iteration")
    savefig(p, joinpath(figure_results_folder, "residual_cost_$(file_name_base).pdf"))

    local u_fb = [Systems.u(fb_res, i, N) for i ∈ 0:N]
    local u_rk_fb = [Systems.u(rk4_res, i, N) for i ∈ 0:N]

    local control_plot = plot(u_fb, label="forward backward gradient method", lw=3) ; plot!(u_rk_fb, label="RK4 gradient method", lw=3) ; ylabel!("uₜ") ; xlabel!("t")
    savefig(control_plot, joinpath(figure_results_folder, "control_plot_$(file_name_base).pdf"))

    local H_fb = [
        forward_backward_sweep_module.H(
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

    local H_rk4_fb = [
        forward_backward_sweep_module.H(
            yₙ = vcat(Systems.x(rk4_res, i, N, x₀), Systems.v(rk4_res, i, N)),
            pₙ = vcat(Systems.λ(rk4_res, i, N), Systems.μ(rk4_res, i, N)),
            uₙ = Systems.u(rk4_res, i, N),
            k = k,
            l₀ = l₀,
            m = m,
            α = α,
            x_d = x_d
        ) 
        for i ∈ 0:N
    ]

    hamoltonian_plot = plot(H_fb, lw = 3, label="Forward Backward", xlabel="t", ylabel = "Hₜ") ; plot!(H_rk4_fb, lw = 3, label= "RK4 Forward Backward") 
    savefig(hamoltonian_plot, joinpath(figure_results_folder, "hamoltonian_plot_$(file_name_base).pdf"))


    local results = Dict{Symbol, Any}()
    results[:rk4_time] = rk4_time
    results[:rk4_iterations] = length(rk4_cost)
    results[:rk4_costs] = rk4_cost
    results[:rk4_results] = rk4_res
    results[:fb_time] = fb_time
    results[:fb_iterations] = length(fb_cost)
    results[:fb_costs] = fb_cost
    results[:fb_results] = fb_res
    
    # Progress bar update (serialize UI-ish calls)
    lock(result_lock) do
        push!(rows, results)
        update!(comp_job); render(pbar)

        Remote_Status_Notifier.send_message(Dict(
            :progress => "$(length(rows) / n_combinations * 100)%",
            :message => "finished iteration for $(file_name_base)",
            :rk4_time => results[:rk4_time],
            :rk4_iterations => results[:rk4_iterations],
            :fb_time => results[:fb_time],
            :fb_iterations => results[:fb_iterations],
        ))
    end
end

Remote_Status_Notifier.send_message(Dict(
    progress => "100%",
    message => "Completed"
))
df = DataFrame(rows)
CSV.write(joinpath(result_folder, output_file_name))


