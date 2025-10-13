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


records = keys(func_results)
record_pairs = [record for record ∈ Iterators.product(records, records) if record[1][1] == record[2][1] && record[1][2] == record[2][2] && record[1][3] == "Modified $(record[2][3])" && func_results[record[1]] != [] && func_results[record[2]] != []]

l₀, x_d, x₀ =  √2, Float64[0,4], Float64[1,1]
T, t₀ = 60.0, 0.0

for record_pair ∈ record_pairs
  modified_info = record_pair[1]
  info = record_pair[2]

  func_result = func_results[info]
  func_result_modified = func_results[modified_info]

  control_modified = [ Systems.u(func_results[modified_info], i, modified_info[1]) for i ∈ 0:modified_info[1] - 1]
  control = [ Systems.u(func_results[info], i, info[1]) for i ∈ 0:info[1] - 1]
  abs_diff = [abs(control_modified[i] - control[i]) for i ∈ 1:length(control)]

  positions = [Systems.x(func_result, i, info[1]) for i ∈ 0:info[1] - 1]
  positions_modified = [Systems.x(func_result_modified, i, modified_info[1]) for i ∈ 0:modified_info[1] - 1]

  x, y = [p[1] for p ∈ positions], [p[2] for p ∈ positions]
  x_modified, y_modified = [p[1] for p ∈ positions_modified], [p[2] for p ∈ positions_modified]

  t = LinRange(t₀,T,length(control))
  control_comparisson = plot(
    t,
    control,
    xlabel="Time (s)", 
    ylabel="Control\nu(t)", 
    title=@sprintf("Control Over Time\n(N=%d, α=%.2f)\nx₀=[%.2f, %.2f] l₀=√2", info[1], info[2], x₀[1], x₀[2]),
    label=info[3],
    dpi = 600,
    titlefont=font(20),   # make title a bit bigger
    top_margin=10mm,        # spacing between title and plot
    legend = :bottomright, 
    lw = 3, 
    color = :blue,
    legendfontsize = 10, 
  )

  plot!(control_comparisson,
    t,
    control_modified,
    color = :red,
    lw = 3,
    label = modified_info[3],
    fillrange = control,
    fillalpha = 0.25,
    ls = :dot
  )


  log_error = plot(
    t,
    log10.(abs_diff .+ eps()),  # add eps to avoid log10(0)
    xlabel="Time (s)", 
    ylabel="log₁₀(|u(t) - u_modified(t)|)",
    title=@sprintf("Logarithmic Error Between Controls\n(N=%d, α=%.2f, %s)\nx₀=[%.2f, %.2f] l₀=√2", info[1], info[2], info[3], x₀[1], x₀[2]),
    label="control error",
    dpi = 600,
    titlefont=font(20),   # make title a bit bigger
    top_margin=10mm,        # spacing between title and plot
    legend = :topright,
    lw = 3, 
    color = :green,
    legendfontsize = 10,
  )

  #display(control_comparisson)
  #display(log_error)

  final_plot = plot(
    control_comparisson,
    log_error,
    layout = (2,1),
    size = (800, 800),
    dpi = 600,
  )
  #display(final_plot)

  position_over_time_plot = quiver(
    x[1:end-1],
    y[1:end-1], 
    quiver=(diff(x),diff(y)), 
    color=:blue, 
    label="$(info[3]) Direction", 
    title=@sprintf("Position Over Time\n(N=%d, α=%.2f)\nx₀=[%.2f, %.2f] l₀=√2", info[1], info[2], Systems.x₀[1], Systems.x₀[2]),
    xlabel="X Position", 
    ylabel="Y Position",
    aspect_ratio= 1,
    dpi=600,
    legend=:bottomleft, 
    arrow=:arrow, 
    arrowsize=0.1, 
    titlefont=font(20),   # make title a bit bigger
    top_margin=10mm,        # spacing between title and plot
    lw = 1, 
    legendfontsize = 8,
  )

  quiver!(
    position_over_time_plot,
    x_modified[1:end-1],
    y_modified[1:end-1], 
    quiver=(diff(x_modified),diff(y_modified)), 
    color=:red, 
    label="$(modified_info[3]) Direction", 
    lw = 1,
    arrow = :arrow,
    ls = :dash
  )

  scatter!(position_over_time_plot, [x_d[1]], [x_d[2]], color = :orange, label = "Desired Position", markershape = :xcross, markersize = 8)
  scatter!(position_over_time_plot, [x₀[1]], [x₀[2]], color = :orange, label = "Starting Position", markersize = 5)
  # Add dummy scatter points for legend
  plot!(
      position_over_time_plot,
      [NaN], [NaN],
      color = :blue,
      label = "$(info[3]) Direction",
      markerstrokewidth = 0,
  )
  plot!(
      position_over_time_plot,
      [NaN], [NaN],
      color = :red,
      label = "$(modified_info[3]) Direction",
      markerstrokewidth = 0,
  )

  #display(position_over_time_plot)

  final_plot = plot(
    control_comparisson,
    position_over_time_plot,
    layout = (2,1),
    size = (800, 800),
    dpi = 600,
  )

  display(final_plot)
  savefig(final_plot, "ThesisPaper/Images/control_and_position_N$(info[1])_alpha$(info[2])_$(info[3])_x0$(x₀)_l0$(round(l₀, digits=2)).pdf")
end