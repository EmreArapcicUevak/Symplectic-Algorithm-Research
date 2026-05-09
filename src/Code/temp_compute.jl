using Pkg; Pkg.instantiate(); Pkg.resolve()
using LinearAlgebra; BLAS.set_num_threads(1)
using InteractiveUtils;

include("../Modules/NewtonMethodModule.jl"); include("../Modules/Systems.jl"); include("../Modules/CLI_Param.jl"); include("../Modules/forward_backward_sweep.jl"); include("../Modules/CLI_Param.jl"); include("../Modules/Settings.jl"); include("../Modules/Remote_Status_Notifier.jl")
using MAT, Serialization, Base.Threads, BenchmarkTools, LinearAlgebra, CSV, DataFrames
using Plots, ArgParse, REPL.TerminalMenus, Printf, ProgressMeter

using Dates

const _BLOCKS = ['▁','▂','▃','▄','▅','▆','▇','█']
function sparkline(xs)
    isempty(xs) && return ""
    lo, hi = extrema(xs)
    hi == lo && return repeat(string(_BLOCKS[end÷2]), length(xs))
    idx = clamp.(round.(Int, (xs .- lo) ./ (hi - lo) .* (length(_BLOCKS)-1)) .+ 1, 1, length(_BLOCKS))
    String(_BLOCKS[idx])
end

fmt_bytes(b) = b < 1<<20 ? (@sprintf "%.0f KiB" b/1024) :
               b < 1<<30 ? (@sprintf "%.1f MiB" b/(1<<20)) :
                           (@sprintf "%.2f GiB" b/(1<<30))

const RUNTIMES   = Float64[]      # rolling per-iter wall time
const HIST_CAP   = 30
const T_START    = time()
const GC_BASE    = Base.gc_num()


param_grid, output_file_name = CLI_Param.get_parameters(sort!(collect(keys(Settings.name_to_func))))
delete!(param_grid, :method)
delete!(param_grid, :educated_guess)

result_lock = ReentrantLock()

keys_ = collect(keys(param_grid))
values_ = [param_grid[key] for key in keys_]
lens = map(length, values_)
n_combinations = reduce(*, lens)
space = CartesianIndices(Tuple(lens))

result_folder = "Results/"
figure_results_folder = joinpath(result_folder, "figure_results/")

mkpath(figure_results_folder)

#pbar = ProgressBar(); comp_job = addjob!(pbar,N = n_combinations, description = "Total Progress")
pbar = Progress(n_combinations; desc = "Total ", showspeed = true, dt = 0.1)
versioninfo(); println("\n", "─"^80, "\n"); flush(stdout)
#start!(pbar); render(pbar)

rows = Vector{Dict{Symbol, Any}}()
Threads.@threads for i ∈ 1:n_combinations
    local tup = ntuple(j -> values_[j][space[i][j]], length(values_))
    local paramaters = NamedTuple{Tuple(keys_)}(tup)

    local N = paramaters[:N]
    local x₀ = paramaters[:x₀]
    local x_d = paramaters[:x_d]
    local m = paramaters[:m]
    local k = paramaters[:k]
    local α = paramaters[:α]

    local l₀ = x_d[2] - m / k
    local v₀ = Float64[0, 0]
    local y₀ = vcat(v₀, x₀)

    local file_name_base = "N=$N,x₀=$(x₀),x_d=$(x_d),m=$(m),k=$(k),α=$α,l₀=$(l₀)"

    local initial_guess = Systems.get_initial_guess(N, x₀, x_d, m, k, l₀)
    local u_guess = [Systems.u(initial_guess, i, N) for i ∈ 0:N]

    local fb_out = @timed forward_backward_sweep_module.forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = y₀, x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-14, max_iter = 500000)
    local fb_time = fb_out[:time]
    local fb_res, fb_cost = fb_out[:value]

    local rk4_out = @timed forward_backward_sweep_module.RK4_forward_backward_sweep(u_guess, 0., 10.; α = α, y₀ = y₀, x_d = x_d, N = N, m = m, k_spring = k, l₀ = l₀, ϵ = 1e-14, max_iter = 500000)
    local rk4_time = rk4_out[:time]
    local rk4_res, rk4_cost = rk4_out[:value]


    local u_fb = [Systems.u(fb_res, i, N) for i ∈ 0:N]
    local u_rk_fb = [Systems.u(rk4_res, i, N) for i ∈ 0:N]


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



    local results = Dict{Symbol, Any}()
    results[:rk4_time] = rk4_time
    results[:rk4_iterations] = length(rk4_cost)
    results[:rk4_costs] = rk4_cost
    results[:rk4_results] = rk4_res
    results[:fb_time] = fb_time
    results[:fb_iterations] = length(fb_cost)
    results[:fb_costs] = fb_cost
    results[:fb_results] = fb_res

    results[:x₀] = x₀
    results[:x_d] = x_d
    results[:m] = m
    results[:k] = k
    results[:α] = α
    results[:l₀] = l₀
    results[:N] = N
    
    # Progress bar update (serialize UI-ish calls)
    lock(result_lock) do
        push!(rows, results)

        push!(RUNTIMES, fb_time + rk4_time)
        length(RUNTIMES) > HIST_CAP && popfirst!(RUNTIMES)

        let
            local done       = length(rows)
            local elapsed    = time() - T_START
            local rate       = done / max(elapsed, eps())                      # iters/sec (real)
            local eta_s      = (n_combinations - done) / max(rate, eps())
            local gcd        = Base.GC_Diff(Base.gc_num(), GC_BASE)
            local load1,_,_  = Sys.loadavg()
            local spark      = sparkline(RUNTIMES)
            local avg_rt     = sum(RUNTIMES)/length(RUNTIMES)

            #update!(comp_job); render(pbar)

            ProgressMeter.next!(pbar; showvalues = [
                (:runtimes,   sparkline(RUNTIMES)),
                (:avg,        @sprintf("%.2fs", avg_rt)),
                (:last,       @sprintf("rk4 %.2fs / fb %.2fs", rk4_time, fb_time)),
                (:mem,        fmt_bytes(Sys.maxrss())),
                (:gc,         @sprintf("%d pauses, %.1fs", gcd.pause, gcd.total_time/1e9)),
                (:load,       @sprintf("%.2f", first(Sys.loadavg()))),
                (:eta,        string(Dates.canonicalize(Dates.Second(round(Int, eta_s))))),
            ])
        end

        local control_plot = plot(u_fb, label="forward backward gradient method", lw=3) ; plot!(control_plot, u_rk_fb, label="RK4 gradient method", lw=3) ; ylabel!("uₜ") ; xlabel!("t")
        savefig(control_plot, joinpath(figure_results_folder, "control_plot_$(file_name_base).pdf"))

        local p =  plot(fb_cost, lw=3, label = "forward backward") ; plot!(p,rk4_cost, lw = 3, label="RK4 forward backward"); ylabel!("Residual Cost") ; xlabel!("Iteration")
        savefig(p, joinpath(figure_results_folder, "residual_cost_$(file_name_base).pdf"))

        local hamoltonian_plot = plot(H_fb, lw = 3, label="Forward Backward", xlabel="t", ylabel = "Hₜ") ; plot!(hamoltonian_plot, H_rk4_fb, lw = 3, label= "RK4 Forward Backward") 
        savefig(hamoltonian_plot, joinpath(figure_results_folder, "hamoltonian_plot_$(file_name_base).pdf"))


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

            Forward-backward
            time:        $(@sprintf("%.2f", results[:fb_time])) s
            iterations:  $(results[:fb_iterations])

            Progress : $(@sprintf("%.2f", length(rows) / n_combinations * 100))%
        """

        Remote_Status_Notifier.send_message(Dict(
            :progress => @sprintf("%.2f%%", length(rows) / n_combinations * 100),
            :message => "finished iteration for $(file_name_base)",
            :rk4_time => results[:rk4_time],
            :rk4_iterations => results[:rk4_iterations],
            :fb_time => results[:fb_time],
            :fb_iterations => results[:fb_iterations],
        ))

        Remote_Status_Notifier.send_ntfy_message(body; title = "Progress Report")
    end
end
#stop!(pbar)

Remote_Status_Notifier.send_message(Dict(
    :progress => "100%",
    :message => "Completed"
))
Remote_Status_Notifier.send_ntfy_message("Computation Complete"; priority = "high", tags = "tada")

new_df = DataFrame(rows)
key_cols = [:N, :m, :k, :α, :l₀, :x₀, :x_d]
jls_path = joinpath(result_folder, "$(output_file_name).jls")

merged_df = if isfile(jls_path)
    old_df = open(deserialize, jls_path)
    vcat(antijoin(old_df, new_df, on = key_cols), new_df; cols = :union)
else
    new_df
end


scalar_cols = [:rk4_time, :rk4_iterations, :fb_time, :fb_iterations, :N, :m, :k, :α, :l₀, :x₀, :x_d] # columns that are scalar values and can be easily saved in CSV
CSV.write(joinpath(result_folder, "$(output_file_name).csv"), select(merged_df, scalar_cols))

open(jls_path, "w") do io
    serialize(io, merged_df)
end

