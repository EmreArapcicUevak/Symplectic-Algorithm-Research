module Remote_Status_Notifier
    using HTTP, JSON3
    include("Settings.jl")

    # in Remote_Status_Notifier.jl
    function send_message(data)
        if isnothing(Settings.HTTP_URL)
            @warn "HTTP_URL not set, skipping HTTP notification"
            return
        end

        try
            HTTP.post(Settings.HTTP_URL, ["Content-Type" => "application/json"], JSON3.write(data))
        catch e
            @warn "notifier failed, continuing" exception=e
        end
    end

    function send_ntfy_message(msg::String; title :: String = "", tags :: String = "", priority :: String = "low")
        if isnothing(Settings.NTFY_TOPIC)
            @warn "NTFY_TOPIC not set, skipping ntfy notification"
            return
        end

        try
            run(`curl -s -o /tmp/ntfy.log --max-time 5 
            -H "Title: $(title)"
            -H "Tags: $(tags)"
            -H "Priority: $(priority)"
            -d $(msg) ntfy.sh/$(Settings.NTFY_TOPIC)`)
        catch e
            @warn "ntfy failed" exception=e
        end
    end
end



