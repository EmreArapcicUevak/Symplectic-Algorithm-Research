include("../Modules/Settings.jl")

using LinearAlgebra; BLAS.set_num_threads(1)
using InteractiveUtils;

include("../Modules/CLI_Param.jl"); include("../Modules/forward_backward_sweep.jl"); include("../Modules/Remote_Status_Notifier.jl"); include("../Modules/ExperimentHarness.jl"); include("../Modules/Systems.jl")
include("../Modules/NewtonMethodModule.jl")
using MAT, Serialization, Base.Threads, BenchmarkTools, LinearAlgebra, CSV, DataFrames
using Plots, Printf


param_grid, output_file_name = CLI_Param.get_parameters(sort!(collect(keys(Settings.name_to_func))))

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

function body(paramaters :: NamedTuple)
    local N = paramaters[:N]
    local x₀ = paramaters[:x₀]
    local x_d = paramaters[:x_d]
    local m = paramaters[:m]
    local k = paramaters[:k]
    local α = paramaters[:α]
    local t₀, T = paramaters[:time_pairs]

    local educated_guess_method_name = paramaters[:educated_guess]
    local method_name = paramaters[:method]

    local l₀ = x_d[2] + m / k
    local v₀ = Float64[0, 0]
    local y₀ = vcat(v₀, x₀)

    local u_guess = randn(Float64, N + 1)

    local educated_guess_method = Settings.educated_guess[educated_guess_method_name]
    local educated_y, educated_cost, g_norm, final_step_lenghts
    local educated_guess_time = -1.0
    educated_guess_time = @elapsed begin
        educated_y, educated_cost, g_norm, final_step_lenghts =
            educated_guess_method(u_guess, t₀, T; α = α, y₀ = y₀, x_d = x_d,
                                  N = N, m = m, k_spring = k, l₀ = l₀,
                                  ϵ = 1e-8, max_iter = 500000)
    end

    if g_norm > 1e-2
        return Dict(:results => nothing, :number_of_iterations => -1,
                    :g_norm => g_norm, :l₀ => l₀,
                    :educated_guess_time => educated_guess_time,
                    :newton_time => -1.0)
    end

    educated_y = contains(lowercase(method_name), "modified") ? educated_y[1:end-1] : educated_y

    local method_function = Settings.name_to_func[method_name]
    local method = x -> method_function(x; N=N, α=α, m=m, k=k, a=Float64[0, -1],
                                        t₀ = t₀, T = T, l₀ = l₀, x₀ = x₀, x_d = x_d)

    local results = nothing
    local newton_time = -1.0
    try
        newton_time = @elapsed begin
            results = NewtonMethodModule.MultiDimentionalNewtonMethod(
                method, x -> NewtonMethodModule.AproximateJacobian(method, x),
                educated_y;
                maxIterations = Settings.NEWTON_MAX_ITER,
                δ = Settings.NEWTON_TOL, ϵ = Settings.NEWTON_TOL)
        end
    catch e
        @warn "Newton method failed" N α method_name e
        results = nothing
        newton_time = -1.0
    end

    return Dict(
        :results => results === nothing ? nothing : results.c,
        :number_of_iterations => results === nothing ? -1 : results.iterations,
        :g_norm => g_norm,
        :l₀ => l₀,
        :educated_guess_time => educated_guess_time,
        :newton_time => newton_time,
    )
end

function result_completed(paramaters, results, rows, progress)
  local N = paramaters[:N]
  local x₀ = paramaters[:x₀]
  local x_d = paramaters[:x_d]
  local m = paramaters[:m]
  local k = paramaters[:k]
  local α = paramaters[:α]
  local t₀, T = paramaters[:time_pairs]
  local method_name = paramaters[:method]
  local educated_guess_name = paramaters[:educated_guess]

  local l₀                    = results[:l₀]
  local g_norm                = results[:g_norm]
  local number_of_iterations  = results[:number_of_iterations]
  local educated_guess_time   = results[:educated_guess_time]
  local newton_time           = results[:newton_time]

  fmt_time(t)  = t < 0 ? "failed" : @sprintf("%.2f s", t)
  fmt_iters(n) = n < 0 ? "failed" : string(n)

  local body = """
          Results for
          N     = $(N)
          x₀    = $(x₀)
          x_d   = $(x_d)
          m     = $(m)
          k     = $(k)
          α     = $(α)
          l₀    = $(@sprintf("%.4f", l₀))
          t₀, T = $(t₀), $(T)

          =============================

          Educated guess — $(educated_guess_name)
          time:        $(fmt_time(educated_guess_time))
          ||Hᵤ||:      $(@sprintf("%.2e", g_norm))

          Newton — $(method_name)
          time:        $(fmt_time(newton_time))
          iterations:  $(fmt_iters(number_of_iterations))

          Progress : $(@sprintf("%.2f", progress * 100))%
      """

  Remote_Status_Notifier.send_ntfy_message(body; title = "Progress Report")
end


res = ExperimentHarness.run_grid(
    body,
    output_file_name;
    param_grid = param_grid,
    key_cols = collect(keys(param_grid)),
    on_iteration = result_completed,
    scalar_cols = [:l₀, :g_norm, :educated_guess_time, :newton_time, :number_of_iterations],
)

if !isnothing(res)
    Remote_Status_Notifier.send_ntfy_message("Experiment completed!"; title = "Completion Notice", priority = "high")
else
    Remote_Status_Notifier.send_ntfy_message("Experiement Interrupted"; title="Completion Notice", priority="high")
end