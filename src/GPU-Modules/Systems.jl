module Systems_GPU

# Change to the path of the library file
const lib = joinpath(@__DIR__, "systems.so")

export SE1
function SE1(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}

    M = length(Y) ÷ (9 * N + 1)
    R = zeros(Float64, (9 * N + 1) * M)


    ccall((:SE1, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
function SE2(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}
    M = length(Y) ÷ (9 * N + 1)
    R = zeros(Float64, (9 * N + 1) * M)

    ccall((:SE2, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
function Modified_SE1(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}
    M = length(Y) ÷ (9 * N)
    R = zeros(Float64, 9 * N * M)

    ccall((:Modified_SE1, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
function Modified_SE2(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}
    M = length(Y) ÷ (9 * N)
    R = zeros(Float64, 9 * N * M)

    ccall((:Modified_SE2, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
function MidPoint(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}
    M = length(Y) ÷ (9 * N + 1)
    R = zeros(Float64, (9 * N + 1) * M)

    ccall((:MidPoint, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
function Modified_MidPoint(Y::Vector{Float64}; N::Integer, α::Float64, m::Float64, k::Float64, a::Vector{Float64},
    t₀::Float64, T::Float64, x₀::Vector{Float64}, l₀::Float64, x_d::Vector{Float64})::Vector{Float64}
    M = length(Y) ÷ (9 * N)
    R = zeros(Float64, 9 * N * M)

    ccall((:Modified_MidPoint, lib), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Clong),
        Y, R, N, α, m, k, a, t₀, T, x₀, l₀, x_d, M
    )

    return R
end
end