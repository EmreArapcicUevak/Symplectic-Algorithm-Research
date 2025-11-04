include("../Modules/NewSystem.jl")
using GLMakie, VideoIO, Serialization, Base.Threads, ColorSchemes, LinearAlgebra
wanted_l₀ = 5.0
wanted_x₀ = Float64[5.3, 2]
wanted_a = Float64[0, -1]
wanted_m = 1.0
wanted_k = 1.0
T = Observable(20.0)
t₀ = Observable(0.0)

# Load data
function round_vector(v::Vector{Float64}, digits::Integer = 2) :: String
    return "[$(join([round(x, digits=digits) for x in v], ", "))]"
end
load_file_name = "results_old_cost_x0$(round_vector(wanted_x₀,2))_l0$(round(wanted_l₀, digits = 2))_m$(round(wanted_m, digits = 2))_k$(round(wanted_k, digits = 2)).jls"
func_number_of_iterations, func_results = deserialize(load_file_name)
function spring_points(A::Point2f, B::Point2f; coils=12, amp=0.08f0, n=200, straight_frac=0.10f0)
    v   = B - A
    L   = LinearAlgebra.norm(v)
    dir = v / L
    perp = Point2f(-dir[2], dir[1])

    t0 = straight_frac * L
    t1 = L - straight_frac * L
    ts = range(0f0, L; length=n)

    pts = Point2f[]
    for t in ts
        base = A + dir * t
        if t < t0 || t > t1
            push!(pts, base)                      # straight ends
        else
            ϕ = 2f0 * π * coils * (t - t0) / (t1 - t0)
            push!(pts, base + perp * (amp * sin(ϕ)))
        end
    end
    return pts
end

# --- 2) Color mapping from stretch/energy to a single color ---
stress_color(ℓ) = get(ColorSchemes.plasma, clamp(abs(ℓ - wanted_l₀)/wanted_l₀, 0, 1))


for (meta_data, func_result) ∈ func_results
  if length(func_result) == 0 continue end
  println("Simulating for $meta_data")
  i = Observable(1)
  playing = Observable(false)
  choosen_N = Observable(meta_data[1])
  choosen_α = Observable(meta_data[2])
  choosen_method = Observable(meta_data[3])
  h = @lift(($T - $t₀)/$choosen_N)


  pendulum_positions = @lift([Point2f(pos) for pos ∈ [Systems.x(func_results[($choosen_N, $choosen_α, $choosen_method)], i, $choosen_N, wanted_x₀) for i ∈ 0:$choosen_N-1]])
  cart_positions = @lift([Point2f(u, 0) for u ∈ [Systems.u(func_results[($choosen_N, $choosen_α, $choosen_method)], i, $choosen_N) for i ∈ 0:$choosen_N-1]])
  Xᵢ = @lift($pendulum_positions[$i])
  Uᵢ = @lift($cart_positions[$i])

  spring_pts = @lift(spring_points($Uᵢ, $Xᵢ))
  curr_len = @lift(LinearAlgebra.norm($Xᵢ - $Uᵢ))


  R = @lift(maximum(norm, $pendulum_positions))
  max_cart_pos = @lift(maximum(norm, $cart_positions))

  spring_color = @lift(stress_color($curr_len))

  f = Figure(size = (1000,700))

  main_axis = Axis(f[1,1], title = "Inverted Pendulum Simulation", xlabel = "X", ylabel = "Y")
  xlims!(main_axis, -R[] - 2, R[] + 2)
  ylims!(main_axis, -R[] - 2, R[] + 2)
  hidespines!(main_axis)

  hlines!(main_axis, [0.0], color = (:black, 0.4))


  # draw pendulum spring
  lines!(main_axis, spring_pts, color = spring_color, linewidth = 3)
  # draw pendulum bob
  scatter!(main_axis, @lift([$Xᵢ]), markersize = 20, color = :orange)

  # draw pivot
  scatter!(main_axis, @lift([$Uᵢ]), markersize = 20, color = :gray, marker = :circle)
  scatter!(main_axis, @lift([$Uᵢ]), markersize = 10, color = :white, marker = :xcross)

  # Stress colorbar
  Colorbar(
    f[1, 2],
    colormap = ColorSchemes.plasma,
    limits = (0, 0.5),  # your max_stretch value
    label = "|ℓ - l₀| / l₀"
  )

  # Draw lenght of spring arrow
  L = @lift(norm($Xᵢ - $Uᵢ))
  L_vec = @lift(($Xᵢ - $Uᵢ))
  P_vec = @lift(Point2f(-$L_vec[2], $L_vec[1]) / norm($L_vec))
  lenght_offset_amount = 0.2f0

  arrows2d!(main_axis,
    @lift[$Uᵢ + $P_vec * lenght_offset_amount, $Xᵢ + $P_vec * lenght_offset_amount],
    @lift([$L_vec, -$L_vec]),
    color = :red
  )


  lenght_text_label_pos = @lift(($Uᵢ + $Xᵢ)/2 + ($P_vec * (lenght_offset_amount + 0.2f0)))
  lenght_text_label_rotation_angle = @lift(atan($L_vec[2], $L_vec[1]))

  lenght_text_label = textlabel!(main_axis,
    lenght_text_label_pos,
    text = @lift("L = $(round($L, digits=4))"),
    text_rotation = lenght_text_label_rotation_angle,
    fontsize = 14,
    alpha = 0.0,
    text_color = :red
  )

  time_label = Label(f[1,1],
    @lift("Time: $(round($t₀ + $h * ($i - 1), digits=2)) s"),
    halign = :left,
    valign = :top,
    padding = (10,10,10,10),
    fontsize = 25,
    color = :black,
    tellwidth = false,
    tellheight = false
  )

  meta_data_label = Label(f[1,1],
    @lift("N = $($choosen_N)\nα = $($choosen_α)\nMethod = $($choosen_method)\nx₀ = [$(round(wanted_x₀[1], digits = 2)), $(round(wanted_x₀[2], digits = 2))]\nl₀ = $(round(wanted_l₀, digits = 2))\na = [$(round(wanted_a[1], digits = 2)), $(round(wanted_a[2], digits = 2))]"),
    halign = :right,
    valign = :top,
    padding = (10,10,10,10),
    fontsize = 12,
    color = :firebrick,
    tellwidth = false,
    tellheight = false,
    lineheight = 1.2
  )

  pivot_position_label = textlabel!(main_axis,
    @lift($Uᵢ + Point2f(0,  $Xᵢ[2] < 0 ? 0.5 : -0.5)),
    text = @lift("($(round($Uᵢ[1], digits=2)), $(round($Uᵢ[2], digits=2)))"),
  )

  pendulum_positions_label = textlabel!(main_axis,
    @lift($Xᵢ + Point2f(0.7, 0)),
    text = @lift("($(round($Xᵢ[1], digits=2)), $(round($Xᵢ[2], digits=2)))"),
  )

  framerate = max(choosen_N[]/(T[] - t₀[]), 1)          # fps you want
  simulation_file_name = "old_cost_pendulumx0=$(round_vector(wanted_x₀))l0=$(round(wanted_l₀, digits=2))N=$(choosen_N[])α=$(choosen_α[])method=$(choosen_method[]).mp4"
  record(f, "Simulations/$simulation_file_name", 1:choosen_N[]; framerate = framerate) do frame
      i[] = frame
      nothing  # block must return nothing
  end

  println("Saved Simulations/$simulation_file_name")
end