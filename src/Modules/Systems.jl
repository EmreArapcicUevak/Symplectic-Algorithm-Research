module Systems
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

  export SE1
  function SE1(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, x₀ :: Vector{Float64} ,l₀ :: Float64, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N + 1)

    Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uᵢ = u(Y, i, N)

      local Lᵢ = L(xᵢ₊₁, uᵢ)
      local Lᵤ = L(xᵢ, uᵢ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local temp_result = zeros(Float64, 9)
      @inbounds temp_result[1:2] =  begin
        δλ + μᵢ
      end

      @inbounds temp_result[3:4] = begin
        δx - vᵢ₊₁
      end

      @inbounds temp_result[5:6] = begin
        local a₁ = (xᵢ₊₁ - [uᵢ, 0]) 
        δμ + (xᵢ₊₁ - x_d) - (c₁ * a₁ * a₁' + c₂ * Matrix{Float64}(I, 2, 2)) * λᵢ
      end

      @inbounds temp_result[7:8] = begin
        local a₁ = (xᵢ₊₁ - [uᵢ, 0]) 
        δv + c₂ * a₁ - (1/m) * a
      end

      @inbounds temp_result[9] =  begin
        local a₁ = k * l₀ / (m * Lᵤ^3)
        local a₂ = k/m * (1 - l₀/Lᵤ)
        local a₃ = (xᵢ - [uᵢ, 0])
        α * uᵢ - dot(λᵢ ,(-a₁ * dot([1.0, 0.0], a₃)) * a₃ - a₂ * [1, 0])
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    @inbounds result[end] = begin
      local xₙ = x(Y, N, N, x₀)
      local uₙ = u(Y, N, N)
      local λₙ = λ(Y,N,N)
      local Lᵤ = L(xₙ, uₙ)

      local a₁ = k * l₀ / (m * Lᵤ^3)
      local a₂ = k/m * (1 - l₀/Lᵤ)
      local a₃ = (xₙ - [uₙ, 0])
      α * uₙ - dot(λₙ ,(-a₁ * dot([1, 0], a₃)) * a₃ - a₂ * [1, 0])
    end

    return result
  end


  export SE2
  function SE2(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N + 1)

    Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uᵢ = u(Y, i, N)
      local uᵢ₊₁ = u(Y, i + 1, N)
      
      local Lᵢ = L(xᵢ, uᵢ₊₁)
      local Lᵤ = L(xᵢ, uᵢ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local temp_result = zeros(Float64, 9)
      @inbounds temp_result[1:2] = begin
        δλ + μᵢ₊₁
      end

      @inbounds temp_result[3:4] = begin
        δx - vᵢ
      end

      @inbounds temp_result[5:6] =  begin
        local a₁ = (xᵢ - [uᵢ₊₁, 0]) 
        δμ + (xᵢ - x_d) - (c₁ * a₁ * a₁' + c₂ * Matrix{Float64}(I, 2, 2)) * λᵢ₊₁
      end

      @inbounds temp_result[7:8] =  begin
        local a₁ = (xᵢ - [uᵢ₊₁, 0]) 
        δv + c₂ * a₁ - (1/m) * a
      end

      @inbounds temp_result[9] = begin
        local a₁ = k * l₀ / (m * Lᵤ^3)
        local a₂ = k/m * (1 - l₀/Lᵤ)
        local a₃ = (xᵢ - [uᵢ, 0])
        α * uᵢ - dot(λᵢ ,(-a₁ * dot([1, 0], a₃)) * a₃ - a₂ * [1, 0])
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    @inbounds result[end] = begin
      local xₙ = x(Y, N, N, x₀)
      local uₙ = u(Y, N, N)
      local Lᵤ = L(xₙ, uₙ)
      local λₙ = λ(Y,N,N)

      local a₁ = k * l₀ / (m * Lᵤ^3)
      local a₂ = k/m * (1 - l₀/Lᵤ)
      local a₃ = (xₙ - [uₙ, 0])
      α * uₙ - dot(λₙ ,(-a₁ * dot([1, 0], a₃)) * a₃ - a₂ * [1, 0])
    end

    return result
  end

  export Modified_SE1
  function Modified_SE1(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N)
    result_lock = ReentrantLock()

    Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uₜ = u(Y, i, N)

      local Lᵢ = L(xᵢ₊₁, uₜ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)
      local c₃ = (xᵢ₊₁ - [uₜ, 0])

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local temp_result = zeros(Float64, 9)
      temp_result[1:2] = begin
        δλ + μᵢ
      end

      temp_result[3:4] = begin
        δx - vᵢ₊₁
      end

      temp_result[5:6] = begin
        δμ + (xᵢ₊₁ - x_d) - (c₁ * c₃ * c₃' + c₂ * Matrix{Float64}(I, 2, 2)) * λᵢ
      end

      temp_result[7:8] = begin
        δv + c₂ * c₃ - (1/m) * a
      end

      temp_result[9] = begin
        α * uₜ - dot(λᵢ ,(-c₁ * dot([1, 0], c₃)) * c₃ - c₂ * [1, 0])
      end

      lock(result_lock) do
        @inbounds result[9i + 1 : 9i + 9] = temp_result 
      end
    end

    return result
  end

  export Modified_SE2
  function Modified_SE2(Y::Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N)

    Threads.@threads for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local uₜ = u(Y, i, N)

      local Lᵢ = L(xᵢ, uₜ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)
      local c₃ = (xᵢ - [uₜ, 0])

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      temp_result = zeros(Float64, 9)
      @inbounds temp_result[1:2] = begin
        δλ + μᵢ₊₁
      end

      @inbounds temp_result[3:4] = begin
        δx - vᵢ
      end

      @inbounds temp_result[5:6] = begin
        δμ + (xᵢ - x_d) - (c₁ * c₃ * c₃' + c₂ * Matrix{Float64}(I, 2, 2)) * λᵢ₊₁
      end

      @inbounds temp_result[7:8] = begin
        δv + c₂ * c₃ - (1/m) * a
      end

      @inbounds temp_result[9] = begin
        α * uₜ - dot(λᵢ₊₁ ,(-c₁ * dot([1, 0], c₃)) * c₃ - c₂ * [1, 0])
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    return result
  end

  export MidPoint
  function MidPoint(Y :: Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N + 1)

    for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local xₘ = xᵢ + (xᵢ₊₁ - xᵢ) * 0.5
      local vₘ = vᵢ + (vᵢ₊₁ - vᵢ) * 0.5
      local λₘ = λᵢ + (λᵢ₊₁ - λᵢ) * 0.5
      local μₘ = μᵢ + (μᵢ₊₁ - μᵢ) * 0.5

      local uᵢ = u(Y, i, N)
      local uᵢ₊₁ = u(Y, i + 1, N)
      local uₘ = (uᵢ + uᵢ₊₁) / 2
      
      local Lᵢ = L(xₘ, uₘ)
      local Lᵤ = L(xᵢ, uᵢ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h


      local temp_result = zeros(Float64, 9)
      @inbounds temp_result[1:2] = begin
        δλ + μₘ
      end

      @inbounds temp_result[3:4] = begin
        δx - vₘ
      end

      @inbounds temp_result[5:6] = begin
        local a₁ = (xₘ - [uₘ, 0]) 
        δμ + (xₘ - x_d) - (c₁ * a₁ * a₁' + c₂ * Matrix{Float64}(I, 2, 2)) * λₘ
      end

      @inbounds temp_result[7:8] = begin
        local a₁ = (xₘ - [uₘ, 0]) 
        δv + c₂ * a₁ - (1/m) * a
      end

      @inbounds temp_result[9] = begin
        local a₁ = k * l₀ / (m * Lᵤ^3)
        local a₂ = k/m * (1 - l₀/Lᵤ)
        local a₃ = (xᵢ - [uᵢ, 0])
        α * uᵢ - dot(λᵢ ,(-a₁ * dot([1, 0], a₃)) * a₃ - a₂ * [1, 0])
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    @inbounds result[end] = begin
      local xₙ = x(Y, N, N, x₀)
      local uₙ = u(Y, N, N)
      local λₙ = λ(Y,N,N)

      local Lᵤ = L(xₙ, uₙ)
      local a₁ = k * l₀ / (m * Lᵤ^3)
      local a₂ = k/m * (1 - l₀/Lᵤ)
      local a₃ = (xₙ - [uₙ, 0])
      α * uₙ - dot(λₙ ,(-a₁ * dot([1, 0], a₃)) * a₃ - a₂ * [1, 0])
    end

    return result
  end

  export Modified_MidPoint
  function Modified_MidPoint(Y :: Vector{Float64}; N :: Integer, α :: Float64, m :: Float64, k :: Float64, a :: Vector{Float64}, t₀ :: Float64, T :: Float64, l₀ :: Float64, x₀ :: Vector{Float64}, x_d :: Vector{Float64}) :: Vector{Float64}
    h = (T - t₀)/N
    result = zeros(Float64, 9N)

    for i ∈ 0:N-1
      local xᵢ = x(Y, i, N, x₀)
      local vᵢ = v(Y, i, N)
      local λᵢ = λ(Y, i, N)
      local μᵢ = μ(Y, i, N)
      local xᵢ₊₁ = x(Y, i + 1, N, x₀)
      local vᵢ₊₁ = v(Y, i + 1, N)
      local λᵢ₊₁ = λ(Y, i + 1, N)
      local μᵢ₊₁ = μ(Y, i + 1, N)

      local xₘ = xᵢ + (xᵢ₊₁ - xᵢ) * 0.5
      local vₘ = vᵢ + (vᵢ₊₁ - vᵢ) * 0.5
      local λₘ = λᵢ + (λᵢ₊₁ - λᵢ) * 0.5
      local μₘ = μᵢ + (μᵢ₊₁ - μᵢ) * 0.5

      local uₜ = u(Y, i, N)

      local Lᵢ = L(xₘ, uₜ)

      local c₁ = k * l₀ / (m * Lᵢ^3)
      local c₂ = k/m * (1 - l₀/Lᵢ)
      local c₃ = (xₘ - [uₜ, 0]) 

      local δλ = (λᵢ₊₁ - λᵢ)/h
      local δμ = (μᵢ₊₁ - μᵢ)/h
      local δv = (vᵢ₊₁ - vᵢ)/h
      local δx = (xᵢ₊₁ - xᵢ)/h

      local temp_result = zeros(Float64, 9)

      @inbounds temp_result[1:2] = begin
        δλ + μₘ
      end

      @inbounds temp_result[3:4] = begin
        δx - vₘ
      end

      temp_result[5:6] = begin
        δμ + (xₘ - x_d) - (c₁ * c₃ * c₃' + c₂ * Matrix{Float64}(I, 2, 2)) * λₘ
      end

      temp_result[7:8] = begin
        δv + c₂ * c₃ - (1/m) * a
      end

      temp_result[9] = begin
        α * uₜ - dot(λₘ ,(-c₁ * dot([1, 0], c₃)) * c₃ - c₂ * [1, 0])
      end

      @inbounds result[9i + 1 : 9i + 9] = temp_result
    end

    return result
  end

  end