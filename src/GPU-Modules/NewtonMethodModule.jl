module NewtonMethodModule_GPU

export AproximateJacobian
function AproximateJacobian(F::Function, x₀::Vector{Float64}; t=1e-6::Float64)
    x_size = length(x₀)
    M = x_size + 1

    Y = zeros(Float64, M * x_size)
    for i in 0:M-1
        Y[i*x_size+1:(i+1)*x_size] = x₀
    end

    for i in 1:x_size
        Y[i*x_size+i] += t
    end

    R = F(Y)

    R_base = R[1:x_size]

    J = zeros(Float64, x_size, x_size)
    for i in 1:x_size
        J[:, i] = (R[i*x_size+1:(i+1)*x_size] - R_base) / t
    end

    return J
end

export AproximateJacobianCentral
function AproximateJacobianCentral(F::Function, x₀::Vector{T}; t=1e-6::Float64) where T<:Real
    x_size = length(x₀)
    M = 2 * x_size

    Y = zeros(Float64, M * x_size)
    for i in 0:M-1
        Y[i*x_size+1:(i+1)*x_size] = x₀
    end

    for i in 1:x_size
        Y[(2*i-2)*x_size+i] += t
        Y[(2*i-1)*x_size+i] -= t
    end

    R = F(Y)

    J = zeros(Float64, x_size, x_size)

    for i in 1:x_size
        R_plus = R[(2*i-2)*x_size+1:(2*i-1)*x_size]
        R_minus = R[(2*i-1)*x_size+1:2*i*x_size]
        J[:, i] = (R_plus - R_minus) / (2 * t)
    end

    return J
end
end