include("../Modules/NewtonMethodModule.jl")
include("../Modules/Systems.jl")

using Revise, MAT, Term.Progress, Serialization, Base.Threads, BenchmarkTools, Serialization
const record_type = Tuple{Int,Float64,String}
const name_to_func = Dict(
  "SE1" => Systems.SE1,
  "SE2" => Systems.SE2,
  "Modified SE1" => Systems.Modified_SE1,
  "Modified SE2" => Systems.Modified_SE2,
)
const func_names = keys(name_to_func)

function get_initial_guess(N :: Integer, α :: Float64, x₀ :: Vector{Float64}, l₀ :: Float64) :: Vector{Float64}
  file_name = "src/EducatedGuesses/zN$(N)_alpha_$(α)_x0$(x₀)_l0$(round(l₀, digits=2)).mat"
  x₀ = nothing
  if isfile(file_name)
    println("Found file $file_name")
    mat_data = matread(file_name)
    x₀ = mat_data["z"]
    x₀ = [x₀[N+1:end]..., 0.0 ,x₀[begin:N]...]
  else
    println("Did not find file $file_name, generating random initial guess")
    x₀ = randn(Float64, 9 * N + 1)
  end

  return x₀
end


initial_guesses = Dict{Tuple{Integer, Float64, Vector{Float64}, Float64}, Vector{Float64}}()
wanted_x₀, wanted_l₀ = Float64[1,1], √2
N_values, α_values = Int[100, 400, 800], Float64[10.0, 5.0, 1.0, 0.5]

func_number_of_iterations = Dict{record_type, Int}()
func_results = Dict{record_type, Vector{Float64}}()

initial_guess_lock = ReentrantLock()
result_lock = ReentrantLock()
progress_bar_lock = ReentrantLock()

pbar = ProgressBar()
comp_job = addjob!(pbar,N = length(N_values) * length(α_values) * length(func_names), description = "Total Progress")
start!(pbar); render(pbar)

func_number_of_iterations, func_results = deserialize("results_x0[1.0, 1.0]_l01.41_ultra.jls")
prod = collect(Iterators.product(N_values, α_values, func_names))
Base.Threads.@threads for i ∈ eachindex(prod)
  local N₀, α₀, method_name = prod[i]

  local skip = false
  lock(progress_bar_lock) do
    if get(func_number_of_iterations, (N₀, α₀, method_name), nothing) !== nothing
      skip = true
      update!(comp_job); render(pbar)
    end
  end

  if skip continue end

  local x₀
  lock(initial_guess_lock) do
    x₀ = get(initial_guesses, (N₀, α₀, wanted_x₀, wanted_l₀), nothing)
    if x₀ === nothing
      x₀ = get_initial_guess(N₀, α₀, wanted_x₀, wanted_l₀)
      initial_guesses[(N₀, α₀, wanted_x₀, wanted_l₀)] = x₀
    end
  end

  local method = x -> name_to_func[method_name](x; N = N₀, α = α₀ ,m = 1. ,k = 1. ,a = Float64[0,1], t₀ = 0.0, T = 60.0, l₀ = wanted_l₀, x_d = Float64[0, 4])
  local x₀ₘ = occursin("Modified", method_name) ? x₀[begin:end-1] : x₀

  local results
  
  try
    results = NewtonMethodModule.MultiDimentionalNewtonMethod(method, x -> NewtonMethodModule.AproximateJacobian(method, x), x₀ₘ; maxIterations = 250, δ = 0.5e-10, ϵ = 0.5e-10)
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

save_file_name = "results_x0$(wanted_x₀)_l0$(round(wanted_l₀, digits=2))_ultra.jls"
println("Saving results to $save_file_name"); serialize(save_file_name, (func_number_of_iterations, func_results))
