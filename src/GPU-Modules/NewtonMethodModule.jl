module NewtonMethodModule_GPU

# Change to the path of the library file
const lib = joinpath(@__DIR__, "systems.so")

@enum SystemsMethod begin
    SE1 = 0
    SE2 = 1
    Modified_SE1 = 2
    Modified_SE2 = 3
    MidPoint = 4
    Modified_MidPoint = 5
end

export AproximateJacobian
function AproximateJacobian(params::Dict, x₀::Vector{Float64}; t=1e-6::Float64)

    x_size = 9 * params[:N] + 1
    _method = string(nameof(params[:method]))
    local method

    if (_method == "SE1")
        method = SE1
    elseif (_method == "SE2")
        method = SE2
    elseif (_method == "Modified_SE1")
        method = Modified_SE1
    elseif (_method == "Modified_SE2")
        method = Modified_SE2
    elseif (_method == "MidPoint")
        method = MidPoint
    elseif (_method == "Modified_MidPoint")
        method = Modified_MidPoint
    else
        error("Unknown method given : $_method")
    end

    if (method == Modified_SE1 || method == Modified_SE2 || method == Modified_MidPoint)
        x_size -= 1
    end

    J = zeros(Float64, x_size, x_size)

    ccall((:ApproximateJacobian, lib), Cvoid, (Cint, Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Cdouble),
        Cint(method), x₀, J, params[:N], params[:α], params[:m], params[:k], params[:a], params[:t₀],
        params[:T], params[:x₀], params[:l₀], params[:x_d], t)

    return J

end

export AproximateJacobianCentral
function AproximateJacobianCentral(params::Dict, x₀::Vector{T}; t=1e-6::Float64) where T<:Real

    x_size = 9 * params[:N] + 1
    _method = string(nameof(params[:method]))
    local method

    if (_method == "SE1")
        method = SE1
    elseif (_method == "SE2")
        method = SE2
    elseif (_method == "Modified_SE1")
        method = Modified_SE1
    elseif (_method == "Modified_SE2")
        method = Modified_SE2
    elseif (_method == "MidPoint")
        method = MidPoint
    elseif (_method == "Modified_MidPoint")
        method = Modified_MidPoint
    else
        error("Unknown method given : $_method")
    end

    if (method == Modified_SE1 || method == Modified_SE2 || method == Modified_MidPoint)
        x_size -= 1
    end

    J = zeros(Float64, x_size, x_size)

    ccall((:ApproximateJacobianCentral, lib), Cvoid, (Cint, Ptr{Float64}, Ptr{Float64}, Clong, Cdouble, Cdouble, Cdouble, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64},
            Cdouble, Ptr{Float64}, Cdouble),
        Cint(method), x₀, J, params[:N], params[:α], params[:m], params[:k], params[:a], params[:t₀],
        params[:T], params[:x₀], params[:l₀], params[:x_d], t)

    return J
end
end
