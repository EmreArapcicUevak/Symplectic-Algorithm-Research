using Plots, Plots.Measures, Serialization, Base.Threads, ColorSchemes, Printf
include("../Modules/NewtonMethodModule.jl")
include("../Modules/NewSystem.jl")
func_number_of_iterations, func_results = deserialize("Grid_Search.jls")



const EntryType = NamedTuple{
    (:N, :α, :m, :l₀, :k, :method, :x_d),
    Tuple{Int, Float64, Float64, Float64, Float64, String, Vector{Float64}}
}

const EntryDataType = Vector{NamedTuple{
  (:x₀, :iterations),
  Tuple{Vector{Float64}, Int}
}}

iteration_data = Dict{EntryType, EntryDataType}()

for (entry, num_of_iter) ∈ func_number_of_iterations
  local key = NamedTuple{(:N, :α, :m, :l₀, :k, :method, :x_d)}(entry)
  if get(iteration_data, key, nothing) === nothing
    iteration_data[key] = Vector{NamedTuple{(:x₀, :iterations), Tuple{Vector{Float64}, Int}}}()
  end
  push!(iteration_data[key], (x₀ = entry[:x₀], iterations = num_of_iter))
end

for (entry, data) ∈ iteration_data
    # Create a custom gradient: green → yellow → red
    cmap = cgrad([:green, :yellow, :red])

    # Replace negative iterations (-1) with NaN for coloring logic
    iterations = [r.iterations for r in data]
    iterations_colored = map(x -> x == -1 ? NaN : x, iterations)
    x_target, y_target = entry.x_d

    plt = plot(
        xlabel = "X",
        ylabel = "Y",
        title = @sprintf(
            "Iterations to Convergence\nN=%d, α=%.2f, m=%.2f\nl₀=%.2f, k=%.2f\n%s",
            entry.N, entry.α, entry.m, entry.l₀, entry.k, entry.method
        ),
        xlims = (
            minimum([[r.x₀[1] for r ∈ data]..., x_target]) - 0.5,
            maximum([[r.x₀[1] for r ∈ data]..., x_target]) + 0.5
        ),
        ylims = (
            minimum([[r.x₀[2] for r ∈ data]..., y_target]) - 0.5,
            maximum([[r.x₀[2] for r ∈ data]..., y_target]) + 0.5
        ),
        dpi = 600,
        colorbar_title = "Iterations",
        titlefont = font(20),
        top_margin = 10mm,
    )

    for (record, iter_val) ∈ zip(data, iterations_colored)
        if isnan(iter_val)
            # Non-converged case (-1): plot as black
            scatter!(
                plt,
                [record.x₀[1]],
                [record.x₀[2]],
                markercolor = :black,
                markersize = 8,
                label = "",
            )
        else
            scatter!(
                plt,
                [record.x₀[1]],
                [record.x₀[2]],
                marker_z = [iter_val],
                markersize = 8,
                label = "",
                color = cmap,
                clims = (0,150)
            )
        end
    end


    # --- Add crosshair lines through desired point ---
    vline!(plt, [0.0], color = RGBA(0, 0, 0, 0.7), lw = 1.5, linestyle = :dash, label = "")
    hline!(plt, [0.0], color = RGBA(0, 0, 0, 0.7), lw = 1.5, linestyle = :dash, label = "")

    # --- Add marker at target position ---
    scatter!(
        plt,
        [x_target],
        [y_target],
        marker = (:star5, 14, :purple, stroke(2, :white)),
        label = "x_d = [$(round(x_target, digits=2)), $(round(y_target, digits=2))]",
    )

    # --- Rest length vector (spring) ---
    # from (0,0) to (0,l₀)
    plot!(
        plt,
        [0, 0],
        [0, entry.l₀ - entry.m / entry.k],
        arrow = :arrow,
        lw = 3,
        color = :dodgerblue,
        label = "Rest length",
    )

    savefig("Images/Iteration_Plots/Iterations_N$(entry.N)_alpha$(round(entry.α, digits=2))_m$(round(entry.m, digits=2))_l0$(round(entry.l₀, digits=2))_k$(round(entry.k, digits=2))_$(entry.method)_x_d[$(round(x_target, digits=2)), $(round(y_target, digits=2))].pdf")
end