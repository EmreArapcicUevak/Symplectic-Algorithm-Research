# GPU vs CPU microbenchmark for system evaluation and approximate Jacobian
# construction. The Newton solver itself is NOT exercised here — only the two
# building blocks whose cost dominates a Newton step.
#
# Run this with a single Julia thread so run_grid does not parallelise the
# sweep and contaminate the per-row timings. BLAS is left at its default so
# the CPU systems get the full BLAS threadpool:
#
#     JULIA_NUM_THREADS=1 julia --project src/Code/gpu_vs_cpu_benchmark.jl
#
# Output:
#   Computation_Results/gpu_vs_cpu_benchmark.csv   — numeric stats per row
#   Computation_Results/gpu_vs_cpu_benchmark.jls   — full DataFrame
#   Computation_Results/gpu_vs_cpu_benchmark_trials.txt
#       — appended BenchmarkTools.Trial pretty-prints for every row
#
# Caveat: BenchmarkTools' :memory / :allocs and Julia's @allocated only see
# host allocations. Device memory used inside systems.so is invisible — those
# columns reflect only the host-side output buffers (R and J).

include("../Modules/Settings.jl")

using LinearAlgebra, Random, Printf, Statistics, BenchmarkTools, Dates

include("../Modules/ExperimentHarness.jl")
include("../Modules/Remote_Status_Notifier.jl")
include("../Modules/Systems.jl")
include("../Modules/NewtonMethodModule.jl")
include("../GPU-Modules/Systems.jl")
include("../GPU-Modules/NewtonMethodModule.jl")

const CPU_SYSTEMS = Dict(
    "SE1"               => Systems.SE1,
    "SE2"               => Systems.SE2,
    "Modified SE1"      => Systems.Modified_SE1,
    "Modified SE2"      => Systems.Modified_SE2,
    "MidPoint"          => Systems.MidPoint,
    "Modified MidPoint" => Systems.Modified_MidPoint,
)

const GPU_SYSTEMS = Dict(
    "SE1"               => Systems_GPU.SE1,
    "SE2"               => Systems_GPU.SE2,
    "Modified SE1"      => Systems_GPU.Modified_SE1,
    "Modified SE2"      => Systems_GPU.Modified_SE2,
    "MidPoint"          => Systems_GPU.MidPoint,
    "Modified MidPoint" => Systems_GPU.Modified_MidPoint,
)

const CPU_JACOBIANS = Dict(
    "AproximateJacobian"        => NewtonMethodModule.AproximateJacobian,
    "AproximateJacobianCentral" => NewtonMethodModule.AproximateJacobianCentral,
)

const GPU_JACOBIANS = Dict(
    "AproximateJacobian"        => NewtonMethodModule_GPU.AproximateJacobian,
    "AproximateJacobianCentral" => NewtonMethodModule_GPU.AproximateJacobianCentral,
)

# Sweep grid. Edit here to extend / shrink.
const PARAM_GRID = Dict{Symbol,AbstractVector}(
    :N        => [50, 100, 200, 400, 800],
    :method   => ["SE1", "SE2", "Modified SE1", "Modified SE2", "MidPoint", "Modified MidPoint"],
    :jacobian => ["AproximateJacobian", "AproximateJacobianCentral"],
)

const OUTPUT_FILE_NAME = "gpu_vs_cpu_benchmark"
const TRIALS_LOG_PATH  = joinpath(Settings.COMPUTATION_RESULTS_FOLDER,
                                  "$(OUTPUT_FILE_NAME)_trials.txt")

# How long BenchmarkTools is allowed to sample each individual measurement.
# Lower = faster total runtime, higher = tighter distributions. Jacobian
# builds are O(N) more expensive than a single F call, so they get a shorter
# budget by default.
const F_BUDGET_SECONDS = 2.0
const J_BUDGET_SECONDS = 2.0

# Fixed physical parameters — Newton convergence is irrelevant here, we only
# need a valid input shape. Values mirror Test/AccuracyTest.jl.
const FIXED = (
    α   = 1.0,
    m   = 1.0,
    k   = 10.0,
    a   = Float64[0.0, -1.0],
    t₀  = 0.0,
    T   = 20.0,
    x₀  = Float64[1.0, 1.0],
    x_d = Float64[2.0, 0.0],
)

const SEED = 0xBEEF

# Pull the numeric distribution out of a BenchmarkTools.Trial.
# All times are in nanoseconds.
function trial_stats(t::BenchmarkTools.Trial)
    times = t.times
    gcs   = t.gctimes
    return (
        min_ns    = minimum(times),
        median_ns = median(times),
        mean_ns   = mean(times),
        max_ns    = maximum(times),
        std_ns    = length(times) > 1 ? std(times) : 0.0,
        samples   = length(times),
        evals     = t.params.evals,
        gc_min_ns = minimum(gcs),
        gc_mean_ns = mean(gcs),
        memory    = t.memory,
        allocs    = t.allocs,
    )
end

# BenchmarkTools' standard pretty-print, captured to a string.
function trial_string(t::BenchmarkTools.Trial)
    io = IOBuffer()
    show(io, MIME"text/plain"(), t)
    return String(take!(io))
end

# Splat a stats NamedTuple into a Dict with a backend prefix.
function merge_stats!(dest::Dict{Symbol,Any}, prefix::Symbol, stats::NamedTuple)
    for (k, v) in pairs(stats)
        dest[Symbol(prefix, :_, k)] = v
    end
    return dest
end

function body(params::NamedTuple)
    N           = params[:N]
    method_name = params[:method]
    jac_name    = params[:jacobian]

    l₀ = FIXED.x_d[2] + FIXED.m / FIXED.k
    is_modified = startswith(method_name, "Modified")
    y_size = is_modified ? 9N : 9N + 1

    Random.seed!(SEED)
    Y = randn(Float64, y_size)

    F_cpu_raw = CPU_SYSTEMS[method_name]
    F_gpu_raw = GPU_SYSTEMS[method_name]

    function_parameters = Dict(
        :N => N, :α => FIXED.α, :m => FIXED.m, :k => FIXED.k, :a => FIXED.a,
        :t₀ => FIXED.t₀, :T => FIXED.T, :x₀ => FIXED.x₀, :l₀ => l₀, :x_d => FIXED.x_d,
    )

    F_cpu = x -> F_cpu_raw(x; function_parameters...)
    F_gpu = x -> F_gpu_raw(x; function_parameters...)

    # --- system eval -------------------------------------------------------
    R_cpu = F_cpu(Y)   # warmup + reference
    R_gpu = F_gpu(Y)
    F_diff_norm = norm(R_cpu - R_gpu)
    F_diff_max  = maximum(abs.(R_cpu - R_gpu))

    F_cpu_trial = @benchmark $F_cpu($Y) seconds=F_BUDGET_SECONDS
    F_gpu_trial = @benchmark $F_gpu($Y) seconds=F_BUDGET_SECONDS

    # --- approximate Jacobian ---------------------------------------------
    J_cpu_build = CPU_JACOBIANS[jac_name](F_cpu_raw; parameters = function_parameters)
    J_gpu_build = GPU_JACOBIANS[jac_name](F_gpu_raw; params      = function_parameters)

    J_cpu = J_cpu_build(Y)   # warmup + reference
    J_gpu = J_gpu_build(Y)
    J_diff_norm = norm(J_cpu - J_gpu)
    J_diff_max  = maximum(abs.(J_cpu - J_gpu))

    J_cpu_trial = @benchmark $J_cpu_build($Y) seconds=J_BUDGET_SECONDS
    J_gpu_trial = @benchmark $J_gpu_build($Y) seconds=J_BUDGET_SECONDS

    F_cpu_stats = trial_stats(F_cpu_trial)
    F_gpu_stats = trial_stats(F_gpu_trial)
    J_cpu_stats = trial_stats(J_cpu_trial)
    J_gpu_stats = trial_stats(J_gpu_trial)

    out = Dict{Symbol,Any}(
        :y_size       => y_size,
        :F_diff_norm  => F_diff_norm,
        :F_diff_max   => F_diff_max,
        :J_diff_norm  => J_diff_norm,
        :J_diff_max   => J_diff_max,
        # Headline speedups, computed from the min (the most reproducible
        # statistic per BenchmarkTools convention).
        :F_speedup_min    => F_cpu_stats.min_ns    / F_gpu_stats.min_ns,
        :F_speedup_median => F_cpu_stats.median_ns / F_gpu_stats.median_ns,
        :J_speedup_min    => J_cpu_stats.min_ns    / J_gpu_stats.min_ns,
        :J_speedup_median => J_cpu_stats.median_ns / J_gpu_stats.median_ns,
        # The Trial pretty-prints, for the sidecar log.
        :_F_cpu_trial_str => trial_string(F_cpu_trial),
        :_F_gpu_trial_str => trial_string(F_gpu_trial),
        :_J_cpu_trial_str => trial_string(J_cpu_trial),
        :_J_gpu_trial_str => trial_string(J_gpu_trial),
    )
    merge_stats!(out, :F_cpu, F_cpu_stats)
    merge_stats!(out, :F_gpu, F_gpu_stats)
    merge_stats!(out, :J_cpu, J_cpu_stats)
    merge_stats!(out, :J_gpu, J_gpu_stats)
    return out
end

function append_trial_log(params, results)
    open(TRIALS_LOG_PATH, "a") do io
        println(io, "="^88)
        println(io, "N=$(params[:N])  method=$(params[:method])  jacobian=$(params[:jacobian])  ($(Dates.now()))")
        println(io, "y_size=$(results[:y_size])  ‖ΔF‖=$(results[:F_diff_norm])  ‖ΔJ‖=$(results[:J_diff_norm])")
        println(io, "-"^88)
        for (label, key) in (
                ("F  CPU", :_F_cpu_trial_str),
                ("F  GPU", :_F_gpu_trial_str),
                ("J  CPU", :_J_cpu_trial_str),
                ("J  GPU", :_J_gpu_trial_str),
            )
            println(io, "[$label]")
            println(io, results[key])
            println(io)
        end
        flush(io)
    end
end

function build_ntfy_body(params, results, progress)
    header = """
    N        = $(params[:N])
    method   = $(params[:method])
    jacobian = $(params[:jacobian])
    y_size   = $(results[:y_size])

    Speedup (min)     F = $(@sprintf("%.2fx", results[:F_speedup_min]))    J = $(@sprintf("%.2fx", results[:J_speedup_min]))
    Speedup (median)  F = $(@sprintf("%.2fx", results[:F_speedup_median])) J = $(@sprintf("%.2fx", results[:J_speedup_median]))
    ‖ΔF‖             = $(@sprintf("%.2e", results[:F_diff_norm]))   max |ΔF| = $(@sprintf("%.2e", results[:F_diff_max]))
    ‖ΔJ‖             = $(@sprintf("%.2e", results[:J_diff_norm]))   max |ΔJ| = $(@sprintf("%.2e", results[:J_diff_max]))

    Progress: $(@sprintf("%.2f", progress * 100))%
    """

    sections = String[]
    for (label, key) in (
            ("F  CPU", :_F_cpu_trial_str),
            ("F  GPU", :_F_gpu_trial_str),
            ("J  CPU", :_J_cpu_trial_str),
            ("J  GPU", :_J_gpu_trial_str),
        )
        push!(sections, "[$label]\n$(results[key])")
    end

    return header * "\n" * join(sections, "\n\n")
end

function report(params, results, _rows, progress)
    append_trial_log(params, results)

    @printf("[%5.1f%%]  N=%-5d  %-18s  %-26s  F: cpu=%.2es gpu=%.2es x%5.1f  |  J: cpu=%.2es gpu=%.2es x%5.1f  |  ‖ΔF‖=%.1e  ‖ΔJ‖=%.1e\n",
        progress * 100,
        params[:N], params[:method], params[:jacobian],
        results[:F_cpu_min_ns] / 1e9, results[:F_gpu_min_ns] / 1e9, results[:F_speedup_min],
        results[:J_cpu_min_ns] / 1e9, results[:J_gpu_min_ns] / 1e9, results[:J_speedup_min],
        results[:F_diff_norm], results[:J_diff_norm],
    )
    flush(stdout)

    title = "Benchmark N=$(params[:N]) $(params[:method]) / $(params[:jacobian])"
    body  = build_ntfy_body(params, results, progress)
    Remote_Status_Notifier.send_ntfy_message(body; title = title, tags = "chart_with_upwards_trend")
end

if Threads.nthreads() > 1
    @warn "Julia is running with $(Threads.nthreads()) threads. run_grid will parallelise the sweep and the BenchmarkTools timings will be unreliable. Launch with JULIA_NUM_THREADS=1 for clean numbers."
end

# Header in the trial log so concatenated runs are still readable.
open(TRIALS_LOG_PATH, "a") do io
    println(io, "\n\n##### Benchmark run started $(Dates.now()) #####\n")
end

const STAT_SUFFIXES = (:min_ns, :median_ns, :mean_ns, :max_ns, :std_ns,
                       :samples, :evals, :gc_min_ns, :gc_mean_ns, :memory, :allocs)
const STAT_COLS = Symbol[Symbol(p, :_, s)
                         for p in (:F_cpu, :F_gpu, :J_cpu, :J_gpu)
                         for s in STAT_SUFFIXES]

res = ExperimentHarness.run_grid(
    body,
    OUTPUT_FILE_NAME;
    param_grid  = PARAM_GRID,
    key_cols    = collect(keys(PARAM_GRID)),
    on_iteration = report,
    scalar_cols = vcat(
        Symbol[:y_size,
               :F_diff_norm, :F_diff_max, :J_diff_norm, :J_diff_max,
               :F_speedup_min, :F_speedup_median,
               :J_speedup_min, :J_speedup_median],
        STAT_COLS,
    ),
)

if !isnothing(res)
    println("\nDone. CSV / JLS in Computation_Results/. Per-row Trial pretty-prints in $(basename(TRIALS_LOG_PATH)).")
    Remote_Status_Notifier.send_ntfy_message("GPU vs CPU benchmark completed.";
        title = "Benchmark Completion", priority = "high")
else
    println("\nInterrupted. Partial results preserved in the .tmp checkpoint.")
    Remote_Status_Notifier.send_ntfy_message("GPU vs CPU benchmark interrupted.";
        title = "Benchmark Interrupted", priority = "high")
end
