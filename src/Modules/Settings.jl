module Settings 
    include("Systems.jl")

    const name_to_func = Dict(
        "SE1" => Systems.SE1,
        "SE2" => Systems.SE2,
        "Modified SE1" => Systems.Modified_SE1,
        "Modified SE2" => Systems.Modified_SE2,
        "MidPoint" => Systems.MidPoint,
        "Modified MidPoint" => Systems.Modified_MidPoint,
    )
end