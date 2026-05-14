include("../Modules/Settings.jl")

using GLMakie, VideoIO, Serialization, Base.Threads, ColorSchemes, LinearAlgebra, DataFrames, Term.Progress
import Term.Progress as Progress
include("../Modules/CLI_Param.jl") ; include("../Modules/Systems.jl")
using CairoMakie
CairoMakie.activate!()

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
stress_color(ℓ, l₀) = get(ColorSchemes.plasma, clamp(abs(ℓ - l₀)/l₀, 0, 1))

# Force a perpendicular vector to point upward, so labels stay on the same side of the spring.
flip_up(v::Point2f) = v[2] < 0 ? -v : v

function round_vector(v::Vector{Float64}, digits::Integer = 2) :: String
    return "[$(join([round(x, digits=digits) for x in v], ", "))]"
end

function open_with_default(path)
    if Sys.isapple()
        run(`open $path`)
    elseif Sys.iswindows()
        run(`cmd /c start "" $path`)
    else
        run(`xdg-open $path`)
    end
end


input_files, display_results = CLI_Param.get_input_file()
println("Input files: ", input_files)
println("Display results after finishing: ", display_results)

pbar = ProgressBar(; columns=:detailed)
file_job = addjob!(pbar; N=length(input_files), description = "Processing files")


plock = ReentrantLock()
a = Float64[0, -1]

const forward_backward_simulation_folder = joinpath(Settings.SIMULATIONS_FOLDER, "Forward_Backward_Simulations/")
mkpath(forward_backward_simulation_folder) 


with(pbar) do
  for file in input_files
      df = deserialize(file)
      row_job = addjob!(pbar; N=nrow(df), description="Simulating...", transient=true)

      Threads.@threads for row in eachrow(df)
        local rk4_norm = row[:rk4_g_norm]
        local fb_norm = row[:fb_g_norm]

        if rk4_norm > 1e-5 || fb_norm > 1e-5
          lock(plock) do
            Progress.update!(row_job)
            Progress.render(pbar)
          end
          continue
        end
        
        local l₀ = row[:l₀]
        local x₀ = row[:x₀]
        local x_d = row[:x_d]
        local m = row[:m]
        local k = row[:k]
        local N = row[:N]
        local α = row[:α]
        local t₀ = row[:t₀]
        local T = row[:T]
        
        local h = (T - t₀) / N
        
        local fb_res = row[:fb_results]
        local rk4_res = row[:rk4_results]

        local i = Observable(1)

        local pendulum_positions_euler = [Point2f(Systems.x(fb_res, k, N, x₀)) for k in 0:N]
        local cart_positions_euler     = [Point2f(Systems.u(fb_res, k, N), 0)  for k in 0:N]

        local pendulum_positions_rk4   = [Point2f(Systems.x(rk4_res, k, N, x₀)) for k in 0:N]
        local cart_positions_rk4       = [Point2f(Systems.u(rk4_res, k, N), 0)  for k in 0:N]

        local Xᵢ_euler = @lift(pendulum_positions_euler[$i])
        local Uᵢ_euler = @lift(cart_positions_euler[$i])

        local Xᵢ_rk4 = @lift(pendulum_positions_rk4[$i])
        local Uᵢ_rk4 = @lift(cart_positions_rk4[$i])

        local spring_pts_euler = @lift(spring_points($Uᵢ_euler, $Xᵢ_euler))
        local curr_len_euler = @lift(LinearAlgebra.norm($Xᵢ_euler - $Uᵢ_euler))

        local spring_pts_rk4 = @lift(spring_points($Uᵢ_rk4, $Xᵢ_rk4))
        local curr_len_rk4 = @lift(LinearAlgebra.norm($Xᵢ_rk4 - $Uᵢ_rk4))


        local R = max(maximum(norm, pendulum_positions_euler),
                      maximum(norm, pendulum_positions_rk4))

        local spring_color_euler = @lift(stress_color($curr_len_euler, l₀))
        local spring_color_rk4 = @lift(stress_color($curr_len_rk4, l₀))

        local f = Figure(size = (1000,700))
        local main_axis = Axis(f[1,1], title = "Inverted Pendulum Simulation", xlabel = "X", ylabel = "Y")
        xlims!(main_axis, -R - 2, R + 2)
        ylims!(main_axis, -R - 2, R + 2)
        hidespines!(main_axis)
        hlines!(main_axis, [0.0], color = (:black, 0.4))



        # draw pendulum spring
        lines!(main_axis, spring_pts_euler, color = spring_color_euler, linewidth = 3)
        lines!(main_axis, spring_pts_rk4, color = spring_color_rk4, linewidth = 3)
        # draw pendulum bob
        scatter!(main_axis, @lift([$Xᵢ_euler]), markersize = 20, color = :blue)
        scatter!(main_axis, @lift([$Xᵢ_rk4]), markersize = 20, color = :orange)

        # draw pivot
        scatter!(main_axis, @lift([$Uᵢ_euler]), markersize = 20, color = :gray, marker = :circle)
        scatter!(main_axis, @lift([$Uᵢ_euler]), markersize = 10, color = :white, marker = :xcross)
        scatter!(main_axis, @lift([$Uᵢ_rk4]), markersize = 20, color = :gray, marker = :circle)
        scatter!(main_axis, @lift([$Uᵢ_rk4]), markersize = 10, color = :white, marker = :xcross)

        # draw target position
        scatter!(main_axis, Point2f(x_d...), markersize = 10, color = :red, marker = :xcross)

        ## Stress colorbar
        #Colorbar(
          #f[1, 2],
          #colormap = ColorSchemes.plasma,
          #limits = (0, 0.5),  # your max_stretch value
          #label = "|ℓ - l₀| / l₀"
        #)

        # Draw lenght of spring arrow
        local L_euler = @lift(norm($Xᵢ_euler - $Uᵢ_euler))
        local L_vec_euler = @lift(($Xᵢ_euler - $Uᵢ_euler))
        local P_vec_euler = @lift(flip_up(Point2f(-$L_vec_euler[2], $L_vec_euler[1]) / norm($L_vec_euler)))
        local lenght_offset_amount = 0.05f0 * R

        arrows2d!(main_axis,
          @lift[$Uᵢ_euler + $P_vec_euler * lenght_offset_amount, $Xᵢ_euler + $P_vec_euler * lenght_offset_amount],
          @lift([$L_vec_euler, -$L_vec_euler]),
          color = :red
        )

        local L_rk4 = @lift(norm($Xᵢ_rk4 - $Uᵢ_rk4))
        local L_vec_rk4 = @lift(($Xᵢ_rk4 - $Uᵢ_rk4))
        local P_vec_rk4 = @lift(flip_up(Point2f(-$L_vec_rk4[2], $L_vec_rk4[1]) / norm($L_vec_rk4)))

        arrows2d!(main_axis,
          @lift[$Uᵢ_rk4 + $P_vec_rk4 * lenght_offset_amount, $Xᵢ_rk4 + $P_vec_rk4 * lenght_offset_amount],
          @lift([$L_vec_rk4, -$L_vec_rk4]),
          color = :red
        )

        local lenght_text_label_pos_euler = @lift(($Uᵢ_euler + $Xᵢ_euler)/2 + ($P_vec_euler * 2 * lenght_offset_amount))
        local lenght_text_label_rotation_angle = @lift(atan($L_vec_euler[2], $L_vec_euler[1]))

        local lenght_text_label_euler = textlabel!(main_axis,
          lenght_text_label_pos_euler,
          text = @lift("L = $(round($L_euler, digits=4))"),
          text_rotation = lenght_text_label_rotation_angle,
          fontsize = 14,
          alpha = 0.0,
          text_color = :red
        )

        local lenght_text_label_pos_rk4 = @lift(($Uᵢ_rk4 + $Xᵢ_rk4)/2 + ($P_vec_rk4 * 2 * lenght_offset_amount))
        local lenght_text_label_rotation_angle = @lift(atan($L_vec_rk4[2], $L_vec_rk4[1]))

        local lenght_text_label_rk4 = textlabel!(main_axis,
          lenght_text_label_pos_rk4,
          text = @lift("L = $(round($L_rk4, digits=4))"),
          text_rotation = lenght_text_label_rotation_angle,
          fontsize = 14,
          alpha = 0.0,
          text_color = :red
        )


        local time_label = Label(f[1,1],
          @lift("Time: $(round(t₀ + h * ($i - 1), digits=2)) s"),
          halign = :left,
          valign = :top,
          padding = (10,10,10,10),
          fontsize = 25,
          color = :black,
          tellwidth = false,
          tellheight = false
        )

        
        local meta_data_label = Label(f[1,1],
          "N = $(N)\nα = $(α)\nx₀ = $(round_vector(x₀))\nl₀ = $(round(l₀, digits=2))\na = $(round_vector(a))\n|Hᵤ| = $(round(row[:fb_g_norm], digits=4)) (FB), $(round(row[:rk4_g_norm], digits=4)) (RK4)",
          halign = :right,
          valign = :top,
          padding = (10,10,10,10),
          fontsize = 12,
          color = :firebrick,
          tellwidth = false,
          tellheight = false,
          lineheight = 1.2
        )

        local pivot_position_label_euler = textlabel!(main_axis,
          @lift($Uᵢ_euler + Point2f(lenght_offset_amount,  $Xᵢ_euler[2] < 0 ? 0.5 : -0.5)),
          text = @lift("Euler ($(round($Uᵢ_euler[1], digits=2)), $(round($Uᵢ_euler[2], digits=2)))"),
          text_align = (:left, :center),
        )
        local pivot_position_label_rk4 = textlabel!(main_axis,
          @lift($Uᵢ_rk4 + Point2f(lenght_offset_amount,  $Xᵢ_rk4[2] < 0 ? 0.5 : -0.5)),
          text = @lift("RK4 ($(round($Uᵢ_rk4[1], digits=2)), $(round($Uᵢ_rk4[2], digits=2)))"),
          text_align = (:right, :center),
        )

        local pendulum_positions_label_euler = textlabel!(main_axis,
          @lift($Xᵢ_euler + Point2f(lenght_offset_amount, 0)),
          text = @lift("Euler ($(round($Xᵢ_euler[1], digits=2)), $(round($Xᵢ_euler[2], digits=2)))"),
          text_align = (:left, :center),
        )

        local pendulum_positions_label_rk4 = textlabel!(main_axis,
          @lift($Xᵢ_rk4 - Point2f(lenght_offset_amount, 0)),
          text = @lift("RK4 ($(round($Xᵢ_rk4[1], digits=2)), $(round($Xᵢ_rk4[2], digits=2)))"),
          text_align = (:right, :center),
        )

        target_fps = 60.0
        duration   = T - t₀
        frames_to_render = max(1, round(Int, target_fps * duration))
        step             = max(1, N ÷ frames_to_render)
        frame_indices    = 1:step:N

        simulation_file_name = joinpath(forward_backward_simulation_folder, "simulation_forward_backward_N=$(N),x0=$(round_vector(x₀)),x_d=$(round_vector(x_d)),l0=$(round(l₀, digits=2)),α=$(α).mp4")
        record(f, simulation_file_name, frame_indices; framerate = target_fps) do frame
            i[] = frame
            nothing  # block must return nothing
        end
        
        lock(plock) do
          Progress.update!(row_job)
          Progress.render(pbar)
        end

        display_results && open_with_default(simulation_file_name)


      end

      Progress.update!(file_job); Progress.render(pbar)
  end
end