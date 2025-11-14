module CLI_Param
  using ArgParse, REPL.TerminalMenus
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

  function get_parameters(func_names :: Vector{String})
    s = ArgParseSettings()
    @add_arg_table s begin
        "--minX"
            help = "The lower portion of the interval to consider when trying out different x0 positions"
            arg_type = Float64

        "--minY"
            help = "The lower portion of the interval to consider when trying out different y0 positions"
            arg_type = Float64

        "--maxX"
            help = "The upper portion of the interval to consider when trying out different x0 positions"
            arg_type = Float64

        "--maxY"
            help = "The upper portion of the interval to consider when trying out different y0 positions"
            arg_type = Float64

        "--numX"
            help = "The number of different x0 positions to try from minX to maxX"
            arg_type = Int

        "--numY"
            help = "The number of different y0 positions to try from minY to maxY"
            arg_type = Int

        "--minHeight"
            help = "The lower portion of the interval to consider when trying out different xd positions"
            arg_type = Float64

        "--maxHeight"
            help = "The upper portion of the interval to consider when trying out different xd positions"
            arg_type = Float64

        "--numHeights"
            help = "The number of different xd positions to try from minHeight to maxHeight"
            arg_type = Int

        "--xValues", "-x"
            help = "List of x0 positions to try"
            arg_type = Float64
            nargs = '+'

        "--yValues", "-y"
            help = "List of y0 positions to try"
            arg_type = Float64
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

        "--lValues", "-l"
            help = "List of spring's rest lengths to try"
            arg_type = Float64
            range_tester = x -> x > 0.0
            nargs = '+'

        "--massValues", "-m"
            help = "List of masses to try"
            arg_type = Float64
            range_tester = x -> x > 0.0
            nargs = '+'

        "--NValues", "-N"
            help = "List of number of discretization points to try"
            arg_type = Int
            nargs = '+'
            range_tester = x -> x > 0

        "--alphaValues", "-a"
            help = "List of alpha values to try"
            arg_type = Float64
            nargs = '+'
            range_tester = x -> x > 0.0

        "--methods", "-M"
            help = "List of method names to try"
            arg_type = String
            range_tester = x -> x in func_names
            nargs = '+'

        "--output", "-o"
            help = "Output file name"
            arg_type = String
            default = "Grid_Search"
    end

    clear_terminal()
    parsed_args = parse_args(s)

    x_values = parsed_args["xValues"]
    y_values = parsed_args["yValues"]
    height_values = parsed_args["heightValues"]


    if isempty(x_values)
      minX = parsed_args["minX"]
      maxX = parsed_args["maxX"]
      numX = parsed_args["numX"]

      while minX === nothing || maxX === nothing || numX === nothing
        if minX === nothing
          minX = Simple_Promt.prompt("Enter value for minX:", Float64)
        end

        if maxX === nothing
          maxX = Simple_Promt.prompt("Enter value for maxX (must be greater then $minX):", Float64)
          while maxX <= minX
            println("maxX must be greater than minX ($minX). Please enter again.")
            maxX = Simple_Promt.prompt("Enter value for maxX (must be greater then $minX):", Float64)
          end
        end

        if numX === nothing
          numX = Simple_Promt.prompt("Enter value for numX:", Int)
        end

        clear_terminal()
        print_slider(minX, maxX, "$numX points")
        confirm = Simple_Promt.prompt("Are these values correct? (yes/no)", Bool)
        if !confirm minX = maxX = numX = nothing end
      end

      @assert maxX > minX "maxX must be greater than minX"
      @assert numX > 1 "numX must be greater than 1"
      x_values = collect(LinRange(minX, maxX, numX))
    end

    if isempty(y_values)
      minY = parsed_args["minY"]
      maxY = parsed_args["maxY"]
      numY = parsed_args["numY"]

      while minY === nothing || maxY === nothing || numY === nothing
        
        if minY === nothing
          minY = Simple_Promt.prompt("Enter value for minY:", Float64)
        end

        if maxY === nothing
          maxY = Simple_Promt.prompt("Enter value for maxY (must be greater then $minY):", Float64)
          while maxY <= minY
            println("maxY must be greater than minY ($minY). Please enter again.")
            maxY = Simple_Promt.prompt("Enter value for maxY (must be greater then $minY):", Float64)
          end
        end

        if numY === nothing
          numY = Simple_Promt.prompt("Enter value for numY:", Int)
        end

        clear_terminal()
        print_slider(minY, maxY, "$numY points")
        confirm = Simple_Promt.prompt("Are these values correct? (yes/no)", Bool)
        if !confirm minY = maxY = numY = nothing end
      end

      @assert maxY > minY "maxY must be greater than minY"
      @assert numY > 1 "numY must be greater than 1"
      y_values = collect(LinRange(minY, maxY, numY))
    end

    if isempty(height_values)
      minHeight = parsed_args["minHeight"]
      maxHeight = parsed_args["maxHeight"]
      numHeights = parsed_args["numHeights"]

      while minHeight === nothing || maxHeight === nothing || numHeights === nothing
        if minHeight === nothing
          minHeight = Simple_Promt.prompt("Enter value for minHeight:", Float64)
        end

        if maxHeight === nothing
          maxHeight = Simple_Promt.prompt("Enter value for maxHeight (must be greater then $minHeight):", Float64)
          while maxHeight <= minHeight
            println("maxHeight must be greater than minHeight ($minHeight). Please enter again.")
            maxHeight = Simple_Promt.prompt("Enter value for maxHeight (must be greater then $minHeight):", Float64)
          end
        end

        if numHeights === nothing
          numHeights = Simple_Promt.prompt("Enter value for numHeights:", Int)
        end

        clear_terminal()
        print_slider(minHeight, maxHeight, "$numHeights points")
        confirm = Simple_Promt.prompt("Are these values correct? (yes/no)", Bool)
        if !confirm minHeight = maxHeight = numHeights = nothing end
      end

      @assert maxHeight > minHeight "maxHeight must be greater than minHeight"
      @assert numHeights > 1 "numHeights must be greater than 1"
      height_values = collect(LinRange(minHeight, maxHeight, numHeights))
    end

    selected_methods = parsed_args["methods"]
    if isempty(selected_methods)
      method_select_menu = MultiSelectMenu(func_names)
      choices = request("Select the methods you would like to run:", method_select_menu)
      @assert !isempty(choices) "At least one method must be selected."
      selected_methods = func_names[collect(choices)]
    end

    alpha_values = parsed_args["alphaValues"]
    l_values = parsed_args["lValues"]
    k_values = parsed_args["kValues"]
    mass_values = parsed_args["massValues"]
    N_values = parsed_args["NValues"]

    if isempty(alpha_values)
      alpha_values = Float64[]
      while true
        push!(alpha_values, Simple_Promt.prompt("Enter a value for α:", Float64))
        clear_terminal()
        print_slider(minimum(alpha_values), maximum(alpha_values), "$(length(alpha_values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
      end
    end

    if isempty(l_values)
      l_values = Float64[]
      while true
        push!(l_values, Simple_Promt.prompt("Enter a value for l₀:", Float64))
        clear_terminal()
        print_slider(minimum(l_values), maximum(l_values), "$(length(l_values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
      end
    end

    if isempty(k_values)
      k_values = Float64[]
      while true
        push!(k_values, Simple_Promt.prompt("Enter a value for k:", Float64))
        clear_terminal()
        print_slider(minimum(k_values), maximum(k_values), "$(length(k_values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
      end
    end

    if isempty(mass_values)
      mass_values = Float64[]
      while true
        push!(mass_values, Simple_Promt.prompt("Enter a value for m:", Float64))
        clear_terminal()
        print_slider(minimum(mass_values), maximum(mass_values), "$(length(mass_values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
      end
    end

    if isempty(N_values)
      N_values = Int[]
      while true
        push!(N_values, Simple_Promt.prompt("Enter a value for N:", Int))
        clear_terminal()
        print_slider(minimum(N_values), maximum(N_values), "$(length(N_values)) points")
        Simple_Promt.prompt("Add another value?", Bool) || break
      end
    end

    param_grid = Dict(
      :l₀ => l_values,
      :x_d => [Vector{Float64}([0,y]) for y in height_values],
      :x₀ => [Vector{Float64}([x,y]) for x in x_values for y in y_values],
      :k => k_values,
      :m => mass_values,
      :N => N_values,
      :α => alpha_values,
      :method => selected_methods
    ) 


    clear_terminal()
    return param_grid, "$(parsed_args["output"]).jls"
  end
end