module Remote_Status_Notifier
    include("Settings.jl")
    using HTTP, JSON3, Logging, LoggingExtras

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
            run(`curl -s -o /dev/null --max-time 5 
            -H "Title: $(title)"
            -H "Tags: $(tags)"
            -H "Priority: $(priority)"
            -d $(msg) ntfy.sh/$(Settings.NTFY_TOPIC)`)
        catch e
            @warn "ntfy failed" exception=e
        end
    end

    function ntfy_format(_, args)
        title = "$(args.level) in $(args._module)"

        body = string(args.message)
        if !isempty(args.kwargs)
            body *= "\n\n" * join(("$k = $v" for (k, v) in args.kwargs), "\n")
        end
        body *= "\n\nat $(basename(string(args.file))):$(args.line)"

        priority = args.level >= Logging.Error ? "high"           : "default"
        tags     = args.level >= Logging.Error ? "rotating_light" : "warning"

        send_ntfy_message(body; title=title, tags=tags, priority=priority)
    end

    ntfy_sink = EarlyFilteredLogger(
        log -> log._module !== Remote_Status_Notifier,
        MinLevelLogger(FormatLogger(ntfy_format), Logging.Warn))

    global_logger(TeeLogger(global_logger(), ntfy_sink))
end



