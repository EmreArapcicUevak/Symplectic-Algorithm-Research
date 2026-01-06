using Serialization, CSV, DataFrames
include("../Modules/Systems.jl")

func_number_of_iterations, func_results = deserialize("Results/Grid_Search.jls")

rows = Vector{Dict{Symbol, Any}}()

for (k, v) in func_results
    record = Dict{Symbol, Any}()
    record[:N] = k.N
    record[:alpha] = k.α
    record[:mass] = k.m
    record[:k_spring] = k.k
    record[:method] = k.method
    record[:x_d] = k.x_d
    record[:x0] = k.x₀
    record[:iterations] = func_number_of_iterations[k]
    if record[:iterations] != -1
        record[:final_position] = Systems.x(v, k.N, k.N)
        record[:position_over_time] = [Systems.x(v, i, k.N, k.x₀) for i in 0:k.N] 
    else
        record[:final_position] = [NaN, NaN]
        record[:position_over_time] = NaN
    end
    push!(rows, record)
end

df = DataFrame(rows)

keys(func_results)
CSV.write("Results/results.csv", df)