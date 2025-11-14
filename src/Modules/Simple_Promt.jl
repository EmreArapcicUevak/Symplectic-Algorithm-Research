module Simple_Promt
  export prompt
  function prompt(message::String, T::Type)
    while true
      print("$message ")
      input = readline()
      try
        if T == Bool
          s_lower = lowercase(strip(input))
          if s_lower in ("yes", "y", "true", "1")
              return true
          elseif s_lower in ("no", "n", "false", "0")
              return false
          else
              println("Please enter yes/no.")
          end
        else
            return parse(T, input)
        end
      catch
          println("Invalid input. Expected a $T.")
      end
    end
  end
end