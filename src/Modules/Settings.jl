module Settings 
    include("Systems.jl")
    using DotEnv

    DotEnv.load!() 

    const name_to_func = Dict(
        "SE1" => Systems.SE1,
        "SE2" => Systems.SE2,
        "Modified SE1" => Systems.Modified_SE1,
        "Modified SE2" => Systems.Modified_SE2,
        "MidPoint" => Systems.MidPoint,
        "Modified MidPoint" => Systems.Modified_MidPoint,
    )

    const BATCH_SAVE_SIZE = parse(Int, get(ENV, "BATCH_SAVE_SIZE", "5"))
    const NTFY_TOPIC = get(ENV, "NTFY_TOPIC", nothing)
    const HTTP_URL = get(ENV, "HTTP_URL", nothing)
end