include("../Modules/NewtonMethodModule.jl")
include("../Modules/Systems.jl")

using Revise, MAT, Term.Progress, Serialization, Base.Threads, BenchmarkTools, Serialization
const record_type = Tuple{Int,Float64,String}
const name_to_func = Dict(
  #"SE1" => Systems.SE1,
  #"SE2" => Systems.SE2,
  "Modified SE1" => Systems.Modified_SE1,
  #"Modified SE2" => Systems.Modified_SE2,
  "MidPoint" => Systems.MidPoint,
  #"Modified MidPoint" => Systems.Modified_MidPoint,
)
const func_names = keys(name_to_func)

function get_initial_guess(N :: Integer, α :: Float64, m :: Float64, k :: Float64, x_d :: Vector{Float64}, l₀ :: Float64) :: Vector{Float64}
  @assert length(x_d) == 2

  file_name = "Asrc/EducatedGuesses/zN$(N)_alpha_$(α)_x0_$(Systems.x₀)_m_$(m)_k_$(k).mat"
  x₀ = nothing
  if isfile(file_name)
    println("Found file $file_name")
    mat_data = matread(file_name)
    x₀ = mat_data["z"]
    x₀ = [x₀[N+1:end]..., 0.0 ,x₀[begin:N]...]
  else
    println("Did not find file $file_name, generating random initial guess")
    x₀ = rand(Float64, 9 * N + 1) * 2

    local ϵᵢ = l₀^2 - Systems.x₀[2]^2
    local initial_position_guess = Systems.x₀[1]
    if ϵᵢ >= 0
      local ϵ = sqrt(ϵᵢ)
      diff = Systems.x₀[1] - x_d[1] 
      if diff > 0 initial_position_guess = x_d[1] + ϵ else initial_position_guess = x_d[1] - ϵ end
    end
    x₀[8N + 1 : (17N + 3) ÷ 2] .= initial_position_guess # Guess for initial u
    x₀[(17N + 5) ÷ 2 : end] .= x_d[1] # Guess for final u

    x₀[1:N] .= repeat(Systems.x₀, N ÷ 2) # Change x₀ till xₙ/2
    x₀[N+1:2N] .= repeat(x_d, N ÷ 2)  # Change xₙ/2 till xₙ
  end

  return x₀
end

initial_guesses = Dict{Tuple{Integer, Float64, Vector{Float64}, Float64}, Vector{Float64}}()
wanted_l₀, wanted_k, wanted_m, wanted_x_d = 2.0, 1.0, 1.0, Float64[0, 1]
N_values, α_values = Int[100,200], Float64[10.0]

func_number_of_iterations = Dict{record_type, Int}()
func_results = Dict{record_type, Vector{Float64}}()
func_err_history = Dict{record_type, Vector{Float64}}()

initial_guess_lock = ReentrantLock()
result_lock = ReentrantLock()
progress_bar_lock = ReentrantLock()

pbar = ProgressBar()
comp_job = addjob!(pbar,N = length(N_values) * length(α_values) * length(func_names), description = "Total Progress")
start!(pbar); render(pbar)

prod = collect(Iterators.product(N_values, α_values, func_names))
Base.Threads.@threads for i ∈ eachindex(prod)
  local N₀, α₀, method_name = prod[i]

  local x₀
  lock(initial_guess_lock) do
    x₀ = get(initial_guesses, (N₀, α₀, Systems.x₀, wanted_l₀), nothing)
    if x₀ === nothing
      x₀ = get_initial_guess(N₀, α₀, wanted_m, wanted_k, wanted_x_d, wanted_l₀)
      initial_guesses[(N₀, α₀, Systems.x₀, wanted_l₀)] = x₀
    end
  end

  func_err_history[(N₀, α₀, method_name)] = Float64[]
  local method = x -> name_to_func[method_name](x; N = N₀, α = α₀ ,m = wanted_m ,k = wanted_k ,a = Float64[0,-1], t₀ = 0.0, T = 20.0, l₀ = wanted_l₀, x_d = wanted_x_d)
  local x₀ₘ = occursin("Modified", method_name) ? x₀[begin:end-1] : x₀

  local results
  
  try
    results = NewtonMethodModule.MultiDimentionalNewtonMethod(method, x -> NewtonMethodModule.AproximateJacobian(method, x), x₀ₘ; maxIterations = 250, δ = 0.5e-10, ϵ = 0.5e-10, history = func_err_history[(N₀, α₀, method_name)])
  catch e
      @warn "Newton method failed" N=N₀ α=α₀ method=method_name
      results = nothing
  end

  # Write results safely
  lock(result_lock) do
      if results === nothing
          func_number_of_iterations[(N₀, α₀, method_name)] = -1
          func_results[(N₀, α₀, method_name)] = []
      else
          func_number_of_iterations[(N₀, α₀, method_name)] = results.iterations
          func_results[(N₀, α₀, method_name)] = results.c
      end
  end

  # Progress bar update (serialize UI-ish calls)
  lock(progress_bar_lock) do
      update!(comp_job); render(pbar)
  end
end
stop!(pbar)

func_number_of_iterations
func_err_history

func_results[(100, 10.0, "Modified SE1")]
a = Systems.Modified_SE1(func_results[(100, 10.0, "Modified SE1")]; N = 100, α = 10.0 ,m = wanted_m ,k = wanted_k ,a = Float64[0,-1], t₀ = 0.0, T = 20.0, l₀ = wanted_l₀, x_d = Float64[0, 1])
using LinearAlgebra
LinearAlgebra.norm(a)



using Plots
plot(func_err_history[(100, 10.0, "Modified SE1")])

for (meta_data, err_history) ∈ func_err_history
  N, α, method_name = meta_data
  p = plot(LinRange(1, length(err_history),length(err_history)), err_history; label = "$method_name, N=$N, α=$α", xlabel = "Iteration", ylabel = "Error norm", title = "Convergence history")
  display(p)
end

save_file_name = "results_x0$(Systems.x₀)_l0$(round(wanted_l₀, digits=2))_FINALLY_2.jls"
println("Saving results to $save_file_name"); serialize(save_file_name, (func_number_of_iterations, func_results))
