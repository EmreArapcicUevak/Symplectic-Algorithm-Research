using Pkg; Pkg.instantiate(); Pkg.resolve()
using LinearAlgebra; BLAS.set_num_threads(1)
using InteractiveUtils;

include("../Modules/CLI_Param.jl"); include("../Modules/forward_backward_sweep.jl"); include("../Modules/Settings.jl"); include("../Modules/Remote_Status_Notifier.jl"); include("../Modules/ExperimentHarness.jl"); include("../Modules/Systems.jl")
using MAT, Serialization, Base.Threads, BenchmarkTools, LinearAlgebra, CSV, DataFrames
using Plots, Printf


default(
    #fontfamily       = "Computer Modern",   # matches LaTeX body text; drop if you don't have it
    titlefontsize    = 20,
    guidefontsize    = 20,   # axis labels (xlabel/ylabel)
    tickfontsize     = 18,
    legendfontsize   = 18,
    framestyle       = :box,
    grid             = true,
    gridalpha        = 0.1,
    size             = (900, 550),
    dpi              = 300,
    margin           = 5Plots.mm,
    lw               = 8,
)

result_folder = "Results/"
figure_results_folder = joinpath(result_folder, "figure_results/")
mkpath(figure_results_folder)

function body(paramaters :: NamedTuple)
    local N = paramaters[:N]
    local x₀ = paramaters[:x₀]
    local x_d = paramaters[:x_d]
    local m = paramaters[:m]
    local k = paramaters[:k]
    local α = paramaters[:α]

    local l₀ = x_d[2] - m / k
    local v₀ = Float64[0, 0]
    local y₀ = vcat(v₀, x₀)

    local initial_guess = Systems.get_initial_guess(N, x₀, x_d, m, k, l₀)
    local u_guess = [Systems.u(initial_guess, i, N) for i ∈ 0:N]

    local fb_out = @timed forward_backward_sweep_module.forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = y₀, x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-12, max_iter = 500000)
    local fb_time = fb_out[:time]
    local fb_res, fb_cost, fb_g_norm, fb_step_lenghts = fb_out[:value]

    local rk4_out = @timed forward_backward_sweep_module.RK4_forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = y₀, x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-12, max_iter = 500000)
    local rk4_time = rk4_out[:time]
    local rk4_res, rk4_cost, rk4_g_norm, rk4_step_lenghts = rk4_out[:value]

    local rk4_residual = Systems.Residual_RK4(rk4_res; N=N, x₀=x₀, x_d=x_d, m=m, k=k, l₀=l₀, α=α, a=Float64[0, -1], t₀ = 0., T = 10.)
    local fb_residual = Systems.Residual_Euler(fb_res; N=N, x₀=x₀, x_d=x_d, m=m, k=k, l₀=l₀, α=α, a=Float64[0, -1], t₀ = 0., T = 10.)

    return Dict(
        :rk4_time => rk4_time,
        :rk4_iterations => length(rk4_cost),
        :rk4_costs => rk4_cost,
        :rk4_results => rk4_res,
        :rk4_g_norm => rk4_g_norm,
        :rk4_step_lenghts => rk4_step_lenghts,
        :rk4_residual_norm => norm(rk4_residual),
        :fb_time => fb_time,
        :fb_iterations => length(fb_cost),
        :fb_costs => fb_cost,
        :fb_results => fb_res,
        :fb_g_norm => fb_g_norm,
        :fb_step_lenghts => fb_step_lenghts,
        :fb_residual_norm => norm(fb_residual),
    )
end

function result_completed(paramaters, results, rows, progress)
    local N = paramaters[:N]
    local x₀ = paramaters[:x₀]
    local x_d = paramaters[:x_d]
    local m = paramaters[:m]
    local k = paramaters[:k]
    local α = paramaters[:α]

    local l₀ = x_d[2] - m / k
    local file_name_base = "N=$N,x₀=$(x₀),x_d=$(x_d),m=$(m),k=$(k),α=$α,l₀=$(l₀)"   

    local fb_res, rk4_res = results[:fb_results], results[:rk4_results]
    local fb_cost, rk4_cost = results[:fb_costs], results[:rk4_costs]

    local H_fb = [
        forward_backward_sweep_module.H(
            yₙ = vcat(Systems.v(fb_res, i, N), Systems.x(fb_res, i, N, x₀)),
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
            yₙ = vcat(Systems.v(rk4_res, i, N), Systems.x(rk4_res, i, N, x₀)),
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

    local u_fb = [Systems.u(fb_res, i, N) for i ∈ 0:N]
    local u_rk_fb = [Systems.u(rk4_res, i, N) for i ∈ 0:N]


    local control_plot = plot(LinRange(0., 10., length(u_fb)), u_fb, label="Forward Backward Euler") ; plot!(control_plot, LinRange(0., 10., length(u_rk_fb)), u_rk_fb, label="Forward Backward RK4") ; ylabel!("uₜ") ; xlabel!("t")
    savefig(control_plot, joinpath(figure_results_folder, "control_plot_$(file_name_base).pdf"))

    local p =  plot(LinRange(0., 10., length(fb_cost)), fb_cost, label = "Forward Backward Euler") ; plot!(p, LinRange(0., 10., length(rk4_cost)), rk4_cost, label="Forward Backward RK4"); ylabel!("Residual Cost") ; xlabel!("Iteration")
    savefig(p, joinpath(figure_results_folder, "residual_cost_$(file_name_base).pdf"))

    local hamiltonian_plot = plot(LinRange(0., 10., length(H_fb)), H_fb, label="Forward Backward Euler", xlabel="t", ylabel = "Hₜ") ; plot!(hamiltonian_plot, LinRange(0., 10., length(H_rk4_fb)), H_rk4_fb, label= "Forward Backward RK4") 
    savefig(hamiltonian_plot, joinpath(figure_results_folder, "hamiltonian_plot_$(file_name_base).pdf"))

    local body = """
            Results for
            N = $(N)
            x₀ = $(x₀)
            x_d = $(x_d)
            m = $(m)
            k = $(k)
            α = $(α)
            l₀ = $(l₀)

            =============================

            RK4 forward-backward
            time:        $(@sprintf("%.2f", results[:rk4_time])) s
            iterations:  $(results[:rk4_iterations])
            ||Hᵤ||:      $(@sprintf("%.2e", results[:rk4_g_norm]))
            α_BB:        $(results[:rk4_step_lenghts]) 
            residual:      $(@sprintf("%.2e", results[:rk4_residual_norm]))

            Forward-backward
            time:        $(@sprintf("%.2f", results[:fb_time])) s
            iterations:  $(results[:fb_iterations])
            ||Hᵤ||:       $(@sprintf("%.2e", results[:fb_g_norm]))
            α_BB:        $(results[:fb_step_lenghts])
            residual:      $(@sprintf("%.2e", results[:fb_residual_norm]))

            Progress : $(@sprintf("%.2f", progress * 100))%
        """

    Remote_Status_Notifier.send_ntfy_message(body; title = "Progress Report")
end

param_grid, output_file_name = CLI_Param.get_parameters(sort!(collect(keys(Settings.name_to_func))))
delete!(param_grid, :method)
delete!(param_grid, :educated_guess)

res = ExperimentHarness.run_grid(
    body,
    output_file_name;
    param_grid = param_grid,
    key_cols = collect(keys(param_grid)),
    on_iteration = result_completed,
    scalar_cols = [:rk4_time, :rk4_iterations, :rk4_g_norm, :rk4_step_lenghts, :fb_time, :fb_iterations, :fb_g_norm, :fb_step_lenghts, :rk4_residual_norm, :fb_residual_norm],
)
if !isnothing(res)
    Remote_Status_Notifier.send_ntfy_message("Experiment completed!"; title = "Completion Notice", priority = "high")
else
    Remote_Status_Notifier.send_ntfy_message("Experiement Interrupted"; title="Completion Notice", priority="high")
end