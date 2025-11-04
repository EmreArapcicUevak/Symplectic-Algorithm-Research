using Pkg; Pkg.instantiate()
using InteractiveUtils; versioninfo()
include("../Modules/NewtonMethodModule.jl"); include("../Modules/Systems.jl"); using Revise, MAT, Term.Progress, Serialization, Base.Threads, BenchmarkTools, Serialization, LinearAlgebra

const name_to_func = Dict(
  "SE1" => Systems.SE1,
  #"SE2" => Systems.SE2,
  "Modified SE1" => Systems.Modified_SE1,
  #"Modified SE2" => Systems.Modified_SE2,
  #"MidPoint" => Systems.MidPoint,
  #"Modified MidPoint" => Systems.Modified_MidPoint,
)
const func_names = collect(keys(name_to_func))

param_grid = Dict(
  :l₀ => Float64[1,2.5,5],
  :x_d => [Vector{Float64}([0,y]) for y in LinRange(2,10, 10)],
  :x₀ => [Vector{Float64}([x,y]) for x in LinRange(-2,2,10) for y in LinRange(0.1,2, 5)],
  :k => Float64[1,3,5],
  :m => Float64[1],
  :N => [100],
  :α => Float64[10],
  :method => func_names
)





############################## Other code ########################################




const RecordType = NamedTuple{
    (:N, :α, :m, :l₀, :k, :method, :x_d),
    Tuple{Int, Float64, Float64, Float64, Float64, String, Vector{Float64}}
}

function get_initial_guess(N :: Integer, x₀ :: Vector{Float64}, x_d :: Vector{Float64}, m :: Float64, k :: Float64, l₀ :: Float64) :: Vector{Float64}
  @assert length(x₀) == 2
  # Compute educated starting spring pos
  local temp_f = u -> -k * (norm(x₀ - [u, 0.0]) - l₀) * (x₀[2]) / norm(x₀ - [u, 0.0]) - m
  local educated_u
  try
    if x₀[1] >= x_d[1]
      educated_u = NewtonMethodModule.QuasiNewtonMethod(temp_f, x₀[1], x₀[1] + 0.5 ; maxIterations = 100, δ = 1e-10, ϵ = 1e-10).c
    else
      educated_u = NewtonMethodModule.QuasiNewtonMethod(temp_f, x₀[1], x₀[1] - 0.5 ; maxIterations = 100, δ = 1e-10, ϵ = 1e-10).c
    end
  catch e 
    #@warn "Educated guess computation failed, using 0.0 as guess" x0=x₀
    educated_u = x₀[1]
  end
  local x₀_guess = rand(Float64, 9 * N + 1) * 2
  local stable_height = l₀ - m / k
  local stable_final_pos = Float64[0.0, stable_height]


  x₀_guess[1:N] .= repeat(x₀, N ÷ 2) # Change x₀ till xₙ/2
  x₀_guess[N+1:2N] .= repeat(stable_final_pos, N ÷ 2)  # Change xₙ/2 till xₙ
  x₀_guess[2N+1:2:4N] .= 0.0  # Change x component of v to 0
  x₀_guess[8N+1:end] .= 0.0  # Change u to 0
  x₀_guess[8N+1:(17N + 3) ÷ 2] .= LinRange(educated_u, 0.0, (N+3) ÷ 2)  # Change half of the u's to educated guess

  return x₀_guess
end

func_number_of_iterations = Dict{
  RecordType, 
  NamedTuple{
    (:x₀, :iterations), 
    Tuple{
      Vector{Float64},
      Int
    }
  }
}()
func_results = Dict{RecordType, Vector{Float64}}()
func_err_history = Dict{RecordType, Vector{Float64}}()

initial_guess_lock = ReentrantLock()
result_lock = ReentrantLock()
progress_bar_lock = ReentrantLock()
param_read = ReentrantLock()



keys_ = collect(keys(param_grid))
values_ = [param_grid[key] for key in keys_]

lens = map(length, values_) 
n_combinations = reduce(*, lens)
space = CartesianIndices(Tuple(lens))

pbar = ProgressBar(); comp_job = addjob!(pbar,N = n_combinations, description = "Total Progress")
start!(pbar); render(pbar)
initial_guesses = Dict{Tuple{Integer, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64}, Vector{Float64}}()
Threads.@threads for i ∈ 1:n_combinations
  local I
  lock(param_read) do 
    I = space[i]
  end
  
  local tup = ntuple(j -> values_[j][I[j]], length(values_))
  local params = NamedTuple{Tuple(keys_)}(tup)

  local initial_guess
  lock(initial_guess_lock) do
    init_guess_key = (params.N, params.x₀, params.x_d, params.m, params.k, params.l₀)
    initial_guess = get(initial_guesses, (init_guess_key), nothing)
    #func_err_history[(N₀, α₀, method_name)] = Float64[]
    if initial_guess === nothing
      initial_guess = get_initial_guess(init_guess_key...)
      initial_guesses[init_guess_key] = initial_guess
    end
  end

  local method = x -> name_to_func[params.method](x; N = params.N, α = params.α ,m = params.m ,k =  params.k, a = Float64[0,-1], t₀ = 0.0, T = 20.0, l₀ = params.l₀, x₀ = params.x₀, x_d = params.x_d)
  local initial_guess_m = occursin("Modified", params.method) ? initial_guess[begin:end-1] : initial_guess

  local results
  
  try
    results = NewtonMethodModule.MultiDimentionalNewtonMethod(method, x -> NewtonMethodModule.AproximateJacobian(method, x), initial_guess_m; maxIterations = 150, δ = 0.5e-10, ϵ = 0.5e-10)
  catch e
      #@warn "Newton method failed" N=params.N α=params.α method=params.method e
      results = nothing
  end

  # Write results safely
  lock(result_lock) do
    local key = NamedTuple{(:N, :α, :m, :l₀, :k, :method, :x_d)}(params)
      if results === nothing
          func_number_of_iterations[key] = (x₀ = params.x₀, iterations = -1)
          func_results[key] = []
      else
          func_number_of_iterations[key] =  (x₀ = params.x₀, iterations = results.iterations)
          func_results[key] = results.c
      end
  end

  # Progress bar update (serialize UI-ish calls)
  lock(progress_bar_lock) do
      update!(comp_job); render(pbar)
  end

end
stop!(pbar)

print(func_number_of_iterations)

function round_vector(v::Vector{Float64}, digits::Integer = 2) :: String
    return "[$(join([round(x, digits=digits) for x in v], ", "))]"
end

save_file_name = "Grid_Search.jls"
println("Saving results to $save_file_name"); serialize(save_file_name, (func_number_of_iterations, func_results))
