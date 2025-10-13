using Plots, Plots.Measures, Serialization, Base.Threads, ColorSchemes, Printf
include("../Modules/NewtonMethodModule.jl")
include("../Modules/Systems.jl")

const record_type = Tuple{Int,Float64,String}
const name_to_func = Dict(
  "SE1" => Systems.SE1,
  "SE2" => Systems.SE2,
  "Modified SE1" => Systems.Modified_SE1,
  "Modified SE2" => Systems.Modified_SE2,
  "MidPoint" => Systems.MidPoint,
  "Modified MidPoint" => Systems.Modified_MidPoint
)
const func_names = keys(name_to_func)
func_number_of_iterations, func_results = deserialize("results_x0[1.0, 1.0]_l01.41_ultra.jls")


l₀, x_d, x₀ =  √2, Float64[0,4], Float64[1,1]
T, t₀ = 60.0, 0.0
for (info, num_of_iterations) in func_number_of_iterations
  println("N: $(info[1]), α: $(info[2]), Method: $(info[3]) => Iterations: $num_of_iterations")
  
  cur_func_results = func_results[info]
  if length(cur_func_results) == 0 continue end

  positions = [Systems.x(cur_func_results, i, info[1]) for i ∈ 0:info[1]]
  control = [Systems.u(cur_func_results, i, info[1]) for i ∈ 0:info[1] - (occursin("Modified", info[3]) ? 1 : 0)]

  x = [p[1] for p in positions]
  y = [p[2] for p in positions]

  # Compute direction of movement (dx, dy) between points
  dx = diff(x)
  dy = diff(y)

  # Compute speed (or any metric you want to use for coloring)
  speed = sqrt.(dx.^2 .+ dy.^2)

  # Normalize speed to [0,1] for colormap lookup
  speed_norm = (speed .- minimum(speed)) ./ (maximum(speed) - minimum(speed))

  # Pick a colormap (e.g., viridis)
  colors = get.(Ref(ColorSchemes.viridis), speed_norm)

  l₀_formated = @sprintf("%.2f", l₀)

  position_over_time_plot = quiver(
    x[1:end-1],
    y[1:end-1], 
    quiver=(dx, dy), 
    color=colors, 
    label="Direction", 
    title=@sprintf("Position Over Time\n(N=%d, α=%.2f, %s)\nx₀=[%.2f, %.2f] l₀=√2", info[1], info[2], info[3], Systems.x₀[1], Systems.x₀[2]),
    xlabel="X Position", 
    ylabel="Y Position",
    aspect_ratio= 1,
    dpi=600,
    legend=:topright, 
    arrow=:arrow, 
    arrowsize=0.1, 
    titlefont=font(20),   # make title a bit bigger
    top_margin=10mm,        # spacing between title and plot
    lw = 2, 
    legendfontsize = 10,
    )
  scatter!(position_over_time_plot, [x_d[1]], [x_d[2]], color = :red, label = "Desired Position", markershape = :xcross, markersize = 8)
  savefig(position_over_time_plot, "ThesisPaper/Images/position_over_time_N$(info[1])_alpha$(info[2])_$(info[3])_x0$(x₀)_l0$(l₀_formated).pdf")

  control_plot = plot(
    LinRange(0,T, length(control)),
    control,
    xlabel="t", 
    ylabel="u(t)", 
    title=@sprintf("Control Over Time\n(N=%d, α=%.2f, %s)\nx₀=[%.2f, %.2f] l₀=√2", info[1], info[2], info[3], x₀[1], x₀[2]),
    label="Control u(t)",
    dpi = 600,
    titlefont=font(20),   # make title a bit bigger
    top_margin=10mm,        # spacing between title and plot
    legend = :bottomright, 
    lw = 2, 
    color = :blue,
    legendfontsize = 10,
  )

  savefig(control_plot, "ThesisPaper/Images/control_N$(info[1])_alpha$(info[2])_$(info[3])_x0$(x₀)_l0$(l₀_formated).pdf")

  display(position_over_time_plot)
  display(control_plot)
end



