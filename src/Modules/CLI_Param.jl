module CLI_Param
  include("Settings.jl")
  
  using ArgParse, REPL.TerminalMenus, Serialization
  include("Simple_Promt.jl")


  function print_slider(min_val::Number, max_val::Number, label::AbstractString; width=40)
    # Ensure the label fits
    label = " $label "
    bar_len = max(width - length(label), 2)

    left_len  = bar_len ÷ 2
    right_len = bar_len - left_len

    println("$(min_val) $(repeat('-', left_len))$(label)$(repeat('-', right_len)) $(max_val)")
  end

  function clear_terminal()
    print("\033c")
  end


function prompt_choice_list(parsed_args, key, choices, label)
    selected = parsed_args[key]
    if isempty(selected)
        menu = MultiSelectMenu(choices)
        picks = request("Select $label:", menu)
        @assert !isempty(picks) "At least one $label must be selected."
        selected = choices[collect(picks)]
    end
    return selected
end

function prompt_choice_one(parsed_args, key, choices, label)
    selected = parsed_args[key]
    if selected === nothing
        menu = RadioMenu(choices)
        pick = request("Select $label:", menu)
        @assert pick > 0 "A $label must be selected."
        selected = choices[pick]
    end
    return selected
end

function prompt_range(parsed_args, key_min, key_max, key_num, label)
    lo = parsed_args[key_min]
    hi = parsed_args[key_max]
    n  = parsed_args[key_num]

    while lo === nothing || hi === nothing || n === nothing
        if lo === nothing
            lo = Simple_Promt.prompt("Enter value for $key_min:", Float64)
        end
        if hi === nothing
            hi = Simple_Promt.prompt("Enter value for $key_max (must be greater than $lo):", Float64)
            while hi <= lo
                println("$key_max must be greater than $key_min ($lo). Please enter again.")
                hi = Simple_Promt.prompt("Enter value for $key_max (must be greater than $lo):", Float64)
            end
        end
        if n === nothing
            n = Simple_Promt.prompt("Enter value for $key_num:", Int)
        end

        clear_terminal()
        print_slider(lo, hi, "$n points")
        if !Simple_Promt.prompt("Are these values correct? (yes/no)", Bool)
            lo = hi = n = nothing
        end
    end

    @assert hi > lo "$key_max must be greater than $key_min"
    @assert n > 1 "$key_num must be greater than 1"
    return collect(LinRange(lo, hi, n))
end

function prompt_list(::Type{T}, label) where {T}
    values = T[]
    while true
        push!(values, Simple_Promt.prompt("Enter a value for $label:", T))
        clear_terminal()
        print_slider(minimum(values), maximum(values), "$(length(values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
    end
    return values
end

function get_parameters(func_names::Vector{String})
    s = ArgParseSettings()
    @add_arg_table s begin
        "--minX"
            help = "Lower bound for x0 sweep"
            arg_type = Float64
        "--maxX"
            help = "Upper bound for x0 sweep"
            arg_type = Float64
        "--numX"
            help = "Number of x0 points from minX to maxX"
            arg_type = Int

        "--minY"
            help = "Lower bound for y0 sweep"
            arg_type = Float64
        "--maxY"
            help = "Upper bound for y0 sweep"
            arg_type = Float64
        "--numY"
            help = "Number of y0 points from minY to maxY"
            arg_type = Int

        "--minHeight"
            help = "Lower bound for xd sweep"
            arg_type = Float64
        "--maxHeight"
            help = "Upper bound for xd sweep"
            arg_type = Float64
        "--numHeights"
            help = "Number of xd points from minHeight to maxHeight"
            arg_type = Int

        "--xValues", "-x"
            help = "List of x0 positions (used with --yValues to form a cartesian grid)"
            arg_type = Float64
            nargs = '+'
        "--yValues", "-y"
            help = "List of y0 positions (used with --xValues to form a cartesian grid)"
            arg_type = Float64
            nargs = '+'

        "--x0Pairs", "-X"
            help = "Explicit x0 points as 'x,y' pairs, e.g. -X 1.0,2.0 3.0,4.0 (overrides --xValues/--yValues)"
            arg_type = String
            nargs = '+'

        "--heightValues", "-d"
            help = "List of xd positions to try"
            arg_type = Float64
            nargs = '+'

        "--kValues", "-k"
            help = "List of spring constants to try"
            arg_type = Float64
            range_tester = x -> x > 0.0
            nargs = '+'

        "--massValues", "-m"
            help = "List of masses to try"
            arg_type = Float64
            range_tester = x -> x > 0.0
            nargs = '+'

        "--NValues", "-N"
            help = "List of discretization points to try"
            arg_type = Int
            range_tester = x -> x > 0
            nargs = '+'

        "--alphaValues", "-a"
            help = "List of alpha values to try"
            arg_type = Float64
            range_tester = x -> x > 0.0
            nargs = '+'

        "--methods", "-M"
            help = "List of method names to try"
            arg_type = String
            range_tester = x -> x in func_names
            nargs = '+'

        "--educatedGuess", "-g"
            help = "Initial-guess strategies to try ($(join(Settings.EDUCATED_GUESS_CHOICES, ", ")))"
            arg_type = String
            range_tester = x -> x in Settings.EDUCATED_GUESS_CHOICES
            nargs = '+' 

        "--output", "-o"
            help = "Output file name (without extension)"
            arg_type = String
            default = "Grid_Search"

        "--timePairs", "-t"
            help = "List of (t0, T) pairs as 't0,T' e.g. -t 0.0,10.0 2.0,15.0"
            arg_type = String
            nargs = '+'
            default = ["0.0,10.0"]
            range_tester = x -> begin
                parts = split(x, ",")
                length(parts) == 2 && parse(Float64, parts[1]) < parse(Float64, parts[2])
            end
    end

    clear_terminal()
    parsed_args = parse_args(s)

    # --- x0 points ---------------------------------------------------------
    x0_points = Vector{Float64}[]
    pairs = parsed_args["x0Pairs"]
    if !isempty(pairs)
        for p in pairs
            xs = split(p, ",")
            @assert length(xs) == 2 "Each --x0Pairs entry must be 'x,y', got '$p'"
            push!(x0_points, [parse(Float64, xs[1]), parse(Float64, xs[2])])
        end
    else
        x_values = parsed_args["xValues"]
        y_values = parsed_args["yValues"]
        if isempty(x_values)
            x_values = parsed_args["minX"] === nothing && parsed_args["maxX"] === nothing && parsed_args["numX"] === nothing ?
                prompt_list(Float64, "x0") :
                prompt_range(parsed_args, "minX", "maxX", "numX", "x0")
        end
        if isempty(y_values)
            y_values = parsed_args["minY"] === nothing && parsed_args["maxY"] === nothing && parsed_args["numY"] === nothing ?
                prompt_list(Float64, "y0") :
                prompt_range(parsed_args, "minY", "maxY", "numY", "y0")
        end
        x0_points = [Float64[x, y] for x in x_values for y in y_values]
    end

    # --- xd points ---------------------------------------------------------
    height_values = parsed_args["heightValues"]
    if isempty(height_values)
        height_values = parsed_args["minHeight"] === nothing && parsed_args["maxHeight"] === nothing && parsed_args["numHeights"] === nothing ?
            prompt_list(Float64, "xd") :
            prompt_range(parsed_args, "minHeight", "maxHeight", "numHeights", "xd")
    end

    # --- scalar lists ------------------------------------------------------
    alpha_values = parsed_args["alphaValues"]; isempty(alpha_values) && (alpha_values = prompt_list(Float64, "α"))
    k_values     = parsed_args["kValues"];     isempty(k_values)     && (k_values     = prompt_list(Float64, "k"))
    mass_values  = parsed_args["massValues"];  isempty(mass_values)  && (mass_values  = prompt_list(Float64, "m"))
    N_values     = parsed_args["NValues"];     isempty(N_values)     && (N_values     = prompt_list(Int,     "N"))

    # --- categorical lists -------------------------------------------------
    selected_methods = prompt_choice_list(parsed_args, "methods",        func_names,              "methods to run")
    educated_guesses = prompt_choice_list(parsed_args, "educatedGuess",  Settings.EDUCATED_GUESS_CHOICES,  "initial-guess strategies")


    time_pairs = Tuple{Float64, Float64}[]
    raw_pairs = parsed_args["timePairs"]
    if !isempty(raw_pairs)
        for p in raw_pairs
            parts = split(p, ",")
            t0, T = parse(Float64, parts[1]), parse(Float64, parts[2])
            push!(time_pairs, (t0, T))
        end
    else
        while true
            t0 = Simple_Promt.prompt("Enter t0:", Float64)
            T  = Simple_Promt.prompt("Enter T (must be greater than t0=$t0):", Float64)
            while T <= t0
                println("T must be greater than t0 ($t0). Please enter again.")
                T = Simple_Promt.prompt("Enter T:", Float64)
            end
            push!(time_pairs, (t0, T))
            Simple_Promt.prompt("Add another (t0, T) pair?", Bool) || break
        end
    end
    param_grid = Dict(
        :x_d            => [Float64[0, y] for y in height_values],
        :x₀             => x0_points,
        :k              => k_values,
        :m              => mass_values,
        :N              => N_values,
        :α              => alpha_values,
        :method         => selected_methods,
        :educated_guess => educated_guesses,
        :time_pairs     => time_pairs,
    )

    clear_terminal()
    return param_grid, parsed_args["output"]
  end

  function get_input_file()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input", "-i"
            help = "Input jls file containing data to visualize (must be in Results/Computation_Results/)"
            arg_type = String
            range_tester = x -> endswith(lowercase(x) , ".jls") && isfile(joinpath(Settings.COMPUTATION_RESULTS_FOLDER, x))
            nargs = '+'
            required = true

        "-d"
            help = "Display the results after finishing"
            action = :store_true
            nargs = 0
    end

    clear_terminal()
    parsed_args = parse_args(s)
    parsed_args["input"] = [joinpath(Settings.COMPUTATION_RESULTS_FOLDER, f) for f in parsed_args["input"]]
    return parsed_args["input"], parsed_args["d"]
  end
end