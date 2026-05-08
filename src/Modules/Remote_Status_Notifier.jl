module Remote_Status_Notifier
    using HTTP, JSON3

    # in Remote_Status_Notifier.jl
    function send_message(data, url = "http://167.99.143.133:3000/data")
        try
            HTTP.post(url, ["Content-Type" => "application/json"], JSON3.write(data))
        catch e
            @warn "notifier failed, continuing" exception=e
        end
    end
end

