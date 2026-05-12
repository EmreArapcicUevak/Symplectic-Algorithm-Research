module ExperimentHarness

include("Settings.jl")
using Printf, Dates, DataFrames, Serialization, CSV, Base.Threads, Term.Progress, InteractiveUtils

function load_checkpoint(output_file_name::AbstractString) :: Vector{Dict{Symbol,Any}}
    tmp_path = joinpath("Results/", "$(output_file_name).tmp")
    if isfile(tmp_path)
        return deserialize(tmp_path)
    else
        return Vector{Dict{Symbol,Any}}()
    end
end

function save_checkpoint(output_file_name, rows)
    final = joinpath("Results/", "$(output_file_name).tmp")
    staging = final * ".partial"
    open(staging, "w") do io
        serialize(io, rows)
        flush(io)
        ccall(:fsync, Cint, (Cint,), fd(io))
    end
    mv(staging, final; force=true)
end

function delete_checkpoint(output_file_name :: AbstractString)
    if isfile(joinpath("Results/", "$(output_file_name).tmp"))
        rm(joinpath("Results/", "$(output_file_name).tmp"))
    end
end

# Async signal handler — sets the flag, doesn't do I/O itself
const shutdown  = Threads.Atomic{Bool}(false)   # set by signal handlers, checked in loop
function install_handlers()
    Base.exit_on_sigint(false)   # SIGINT -> InterruptException
    @static if Sys.isunix()
        handler = @cfunction(
            (sig::Cint) -> (shutdown[] = true; nothing),
            Cvoid, (Cint,))
        for sig in (1, 15, 3)    # SIGHUP, SIGTERM, SIGQUIT
            ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), sig, handler)
        end
    end
end


export run_grid
"""
    run_grid(body, output_file_name; kwargs...)

Sweep `body(params)` over the Cartesian product of `param_grid`, in parallel,
with a live progress display, and save the merged results.

# Arguments
- `body(params::NamedTuple) -> Dict{Symbol,Any}` — does the work. Whatever
  it returns is recorded; the parameter values are merged in automatically.

# Keyword arguments
- `param_grid::AbstractDict{Symbol,<:AbstractVector}` — the sweep.
- `key_cols::Vector{Symbol}` — identity columns for merging with prior runs.
- `scalar_cols::Vector{Symbol} = Symbol[]` — columns to also write to CSV.
- `result_folder::AbstractString = "Results/"`.
- `on_iteration::Function = (params, results, rows) -> nothing` — runs
  inside the result lock. Use it for per-iteration plotting / notifications.
- `description::AbstractString = "Total "`.
- `progress_dt::Real = 0.1`.

Returns the merged DataFrame.
"""
function run_grid(body::Function, output_file_name::AbstractString;
    param_grid    :: AbstractDict,
    key_cols      :: AbstractVector{Symbol},
    scalar_cols   :: AbstractVector{Symbol} = Symbol[],
    result_folder :: AbstractString          = "Results/",
    on_iteration  :: Function                = (_,_,_,_) -> nothing,
    description   :: AbstractString          = "Total ",
)
    local keys_   = collect(keys(param_grid))
    local values_ = [param_grid[k] for k in keys_]
    local lens    = map(length, values_)
    local n_comb  = reduce(*, lens)
    local space   = CartesianIndices(Tuple(lens))

    local rows  = load_checkpoint(output_file_name)
    local rlock = ReentrantLock()

    local pbar = ProgressBar(; columns = :detailed);
    local job = addjob!(pbar, N = n_comb, description = description)
    versioninfo(); println("\n", "─"^80, "\n"); flush(stdout)
    start!(pbar); render(pbar)

    install_handlers()
    try
    Threads.@threads for i in 1:n_comb
        shutdown[] && continue

        local tup    = ntuple(j -> values_[j][space[i][j]], length(values_))
        local params = NamedTuple{Tuple(keys_)}(tup)
        local param_pairs = pairs(params)

        if findfirst(d -> issubset(param_pairs, d), rows) !== nothing
            lock(rlock) do 
                update!(job); render(pbar) 
            end
            continue
        end

        local results = body(params)
        # merge param values into the row
        for k in keys_
            results[k] = params[k]
        end

        lock(rlock) do
            push!(rows, results)
            if length(rows) % Settings.BATCH_SAVE_SIZE == 0
                save_checkpoint(output_file_name, rows)
            end

            try
                on_iteration(params, results, rows, length(rows) / n_comb)
            catch err
                @warn "on_iteration failed" exception=(err, catch_backtrace())
            end

            update!(job); render(pbar)
        end
    end
    
    catch e
        @warn "Exception caught during execution: $e"
        if e isa InterruptException
        elseif e isa CompositeException &&
            all(x -> x isa InterruptException ||
                        (x isa TaskFailedException &&
                        x.task.exception isa InterruptException),
                e.exceptions)
        else
            rethrow()
        end
    finally
        save_checkpoint(output_file_name, rows)
    end
    stop!(pbar)

    if shutdown[] || length(rows) < n_comb
        @info "Shutdown signal received. Progress saved."
        return nothing
    end

    delete_checkpoint(output_file_name)
    new_df   = DataFrame(rows)
    jls_path = joinpath(result_folder, "$(output_file_name).jls")

    merged_df = if isfile(jls_path)
        old_df = open(deserialize, jls_path)
        vcat(antijoin(old_df, new_df, on = key_cols), new_df; cols = :union)
    else
        new_df
    end

    if !isempty(scalar_cols)
        CSV.write(joinpath(result_folder, "$(output_file_name).csv"),
                  select(merged_df, scalar_cols..., key_cols...))
    end

    open(jls_path, "w") do io
        serialize(io, merged_df)
    end

    return merged_df
end

end # module