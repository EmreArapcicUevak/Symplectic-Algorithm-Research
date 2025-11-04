
module NewSystems
  using LinearAlgebra, Base.Threads
  L(x :: Vector{Float64}, u :: Float64) :: Float64 = norm(x - [u, 0],2)

  function x(Y :: Vector{Float64}, i :: Integer, N :: Integer, x₀ :: Vector{Float64} = Float64[1, 1]) :: Vector{Float64} 
    @assert i >= 0 && i <= N "Invalid index for x"
    if i == 0
      return copy(x₀)
    else
      local index = 1 + 2 * (i - 1)
      return [Y[index], Y[index+1]]
    end
  end

  function v(Y :: Vector{Float64}, i :: Integer, N :: Integer, v₀ :: Vector{Float64} = Float64[0, 0]) :: Vector{Float64} 
    @assert i >= 0 && i <= N "Invalid index for v"
    if i == 0
      return copy(v₀)
    else
      local index = 2N + 1 + 2 * (i - 1)
      return Float64[Y[index], Y[index+1]]
    end
  end

  function λ(Y :: Vector{Float64}, i :: Integer, N :: Integer, λₙ :: Vector{Float64} = Float64[0, 0]) :: Vector{Float64} 
    @assert i >= 0 && i <= N "Invalid index for λ"
    if i == N
      return copy(λₙ)
    else
      local index = 4N + 1 + 2 * i
      return Float64[Y[index], Y[index+1]]
    end
  end

  function μ(Y :: Vector{Float64}, i :: Integer, N :: Integer, μₙ :: Vector{Float64} = Float64[0, 0]) :: Vector{Float64} 
    @assert i >= 0 && i <= N "Invalid index for μ"
    if i == N
      return copy(μₙ)
    else
      local index = 6N + 1 + 2 * i 
      return Float64[Y[index], Y[index+1]]
    end
  end

  function u(Y :: Vector{Float64}, i :: Integer, N :: Integer) :: Float64
    @assert i >= 0 && i <= N  "Invalid index for u"
    local index = 8N + 1 + i
    return Y[index]
  end

  function SE1(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N + 1)

    Base.Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uᵢ = u(Y, i, N)

      local rᵢ = xᵢ₊₁ - [uᵢ, 0]
      local rᵤ = xᵢ - [uᵢ, 0]
      local Lᵢ = LinearAlgebra.norm(rᵢ, 2)
      local Lᵤ = LinearAlgebra.norm(rᵤ, 2)

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local ∇ₓₓ²U = k * (1 - l₀/Lᵢ) * I + k * l₀ / Lᵢ^3 * (rᵢ * rᵢ')
      local δᵤₓU = k * (-(1 - l₀/Lᵢ) * [1.0, 0.0] - (l₀ * rᵤ[1]) / Lᵤ^3 * rᵤ)

      local temp_result = zeros(Float64, 9)
      temp_result[1:2] =  begin
        δλ + μᵢ
      end

      temp_result[3:4] = begin
        δx - vᵢ₊₁
      end

      temp_result[5:6] = begin
        δμ - [0.0, 1.0] - (1/m) * ∇ₓₓ²U * λᵢ
      end

      temp_result[7:8] = begin
        δv  + k/m * (1 - l₀/Lᵢ) * rᵢ - (1/m) * a
      end

      temp_result[9] =  begin
        α * uᵢ - (1/m) * dot(λᵢ, δᵤₓU)
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    result[end] = begin
      local xₙ = x(Y, N, N, x₀)
      local uₙ = u(Y, N, N)
      local λₙ = λ(Y,N,N)

      local rᵤ = xₙ - [uₙ, 0]
      local Lᵤ = LinearAlgebra.norm(rᵤ, 2)
      local δᵤₓU = k * (-(1 - l₀/Lᵤ) * [1.0, 0.0] - (l₀ * rᵤ[1]) / Lᵤ^3 * rᵤ)

      α * uₙ - (1/m) * dot(λₙ, δᵤₓU)
    end

    return result
  end

  function Modified_SE1(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N)

    Base.Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uᵢ = u(Y, i, N)

      local rᵢ = xᵢ₊₁ - [uᵢ, 0]
      local Lᵢ = LinearAlgebra.norm(rᵢ, 2)

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local ∇ₓₓ²U = k * (1 - l₀/Lᵢ) * I + k * l₀ / Lᵢ^3 * (rᵢ * rᵢ')
      local δᵤₓU = k * (-(1 - l₀/Lᵢ) * [1.0, 0.0] - (l₀ * rᵢ[1]) / Lᵢ^3 * rᵢ)

      local temp_result = zeros(Float64, 9)
      temp_result[1:2] =  begin
        δλ + μᵢ
      end

      temp_result[3:4] = begin
        δx - vᵢ₊₁
      end

      temp_result[5:6] = begin
        δμ - [0.0, 1.0] - (1/m) * ∇ₓₓ²U * λᵢ
      end

      temp_result[7:8] = begin
        δv  + k/m * (1 - l₀/Lᵢ) * rᵢ - (1/m) * a
      end

      temp_result[9] =  begin
        α * uᵢ - (1/m) * dot(λᵢ, δᵤₓU)
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    return result
  end
end