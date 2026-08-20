using JSON: JSON
using Statistics: mean, median, quantile
using Printf: @printf
using ChaoticDynamicalSystemLibrary: ChaoticDynamicalSystemLibrary
using CairoMakie

"""
Precompute maps of concatenated names -> k (2 or 3).

known_names: Vector of the exact strings that appear when you call `string(s)` on each system
             (e.g., `known_names = string.(systems)`).
Returns (concat2::Dict{String,Bool}, concat3::Dict{String,Bool})
"""
function build_concat_maps(known_names::Vector{String})
    concat2 = Dict{String,Bool}()
    concat3 = Dict{String,Bool}()
    N = length(known_names)
    @inbounds for i in 1:N, j in 1:N
        if i != j
            concat2[known_names[i] * known_names[j]] = true
        end
    end
    @inbounds for i in 1:N, j in 1:N, k in 1:N
        if i != j && j != k && i != k
            concat3[known_names[i] * known_names[j] * known_names[k]] = true
        end
    end
    return concat2, concat3
end

"""
Infer k (=2 or 3) from the filename using the precomputed concat maps.
filename should be like ".../fixed_<A><B>.json" or ".../fixed_<A><B><C>.json".
Returns 2, 3, or nothing if not recognized.
"""
function infer_k_from_filename(filepath::AbstractString, concat2::Dict{String,Bool}, concat3::Dict{String,Bool})
    base = basename(filepath)
    if !startswith(base, "fixed_") || !endswith(base, ".json")
        return nothing
    end
    stem = base[7:end-5]  # strips "fixed_" prefix and ".json" suffix
    if haskey(concat3, stem)
        return 3
    elseif haskey(concat2, stem)
        return 2
    else
        return nothing
    end
end

function load_json(fp::AbstractString)
    txt = read(fp, String)

    if Base.find_package("JSON") !== nothing
        @eval using JSON
        try
            return JSON.parsefile(fp; dicttype=Dict{String,Any})
        catch
            return JSON.parse(txt; dicttype=Dict{String,Any})
        end
    end

    if Base.find_package("JSON3") !== nothing
        @eval using JSON3
        return JSON3.read(txt)
    end

    error("No JSON package available. Please `using JSON` or `using JSON3`.")
end


function collect_metrics_by_k(base_dir::AbstractString,
                              known_names::Vector{String})
    concat2, concat3 = build_concat_maps(known_names)

    init_dirs = filter(name -> isdir(joinpath(base_dir, name)), readdir(base_dir))
    out = Dict{String, Dict{Int, Vector}}()

    for init in init_dirs
        init_path = joinpath(base_dir, init)
        json_files = filter(f ->
            startswith(basename(f), "fixed_") && endswith(f, ".json"),
            readdir(init_path; join=true)
        )
        buckets = Dict(
            2 => Vector{NamedTuple{(:file,:systems,:mean,:std), Tuple{String, Any, Float64, Float64}}}(),
            3 => Vector{NamedTuple{(:file,:systems,:mean,:std), Tuple{String, Any, Float64, Float64}}}(),
        )

        for fp in json_files
            k = infer_k_from_filename(fp, concat2, concat3)
            if k ∉ (2, 3)
                @warn "Could not infer k from filename; skipping" file=fp
                continue
            end

            obj = load_json(fp)

            mean_overall = obj["mean_error_overall"]
            std_overall  = obj["std_error_overall"]
            systems      = haskey(obj, "systems") ? obj["systems"] : missing
            push!(buckets[k], (file=basename(fp), systems=systems,
                               mean=mean_overall, std=std_overall))
        end

        out[init] = buckets
    end
    return out
end


function summarize_bucket(metrics_vec)
    if isempty(metrics_vec)
        return Dict(:n_files=>0, :avg_mean=>NaN, :avg_std=>NaN, :best=>nothing, :worst=>nothing)
    end
    means = getfield.(metrics_vec, :mean)
    stds  = getfield.(metrics_vec, :std)
    bi = argmin(means); wi = argmax(means)
    best = metrics_vec[bi]; worst = metrics_vec[wi]
    return Dict(
        :n_files => length(metrics_vec),
        :avg_mean => mean(means),
        :avg_std  => mean(stds),
        :best  => Dict(:mean=>best.mean,  :std=>best.std,  :systems=>best.systems,  :file=>best.file),
        :worst => Dict(:mean=>worst.mean, :std=>worst.std, :systems=>worst.systems, :file=>worst.file),
    )
end

"""
Summarize and print results by init, split by k=2 and k=3 using filename parsing.

known_names must be the exact names used in filenames (e.g., `string.(systems)`).
"""
function summarize_parc_results_split(known_names::Vector{String}, base_dir::AbstractString="parc_results")
    raw = collect_metrics_by_k(base_dir, known_names)
    summaries = Dict{String, Dict{Int, Dict}}()

    println("=== PARC Summary by init, split by number of systems (lower mean = better) ===")
    for init in sort(collect(keys(raw)))
        s2 = summarize_bucket(raw[init][2])
        s3 = summarize_bucket(raw[init][3])
        summaries[init] = Dict(2=>s2, 3=>s3)

        @printf("\n[%s]\n", init)
        for k in (2, 3)
            s = summaries[init][k]
            @printf("  (k=%d) combos: %d\n", k, s[:n_files])
            @printf("        avg mean_overall: %.6f\n", s[:avg_mean])
            @printf("        avg std_overall : %.6f\n", s[:avg_std])
            if s[:best] !== nothing
                b = s[:best]; w = s[:worst]
                println("        best:")
                @printf("          mean=%.6f, std=%.6f, file=%s\n",
                        b[:mean], b[:std], b[:file])
                println("        worst:")
                @printf("          mean=%.6f, std=%.6f, file=%s\n",
                        w[:mean], w[:std], w[:file])
            end
        end
    end

    println("\n=== Inits ranked by avg mean_error_overall (k=2) ===")
    rank2 = sort(collect(summaries), by = x -> x[2][2][:avg_mean])
    for (i, (init, sdict)) in enumerate(rank2)
        s = sdict[2]
        @printf("%2d) %-28s  avg_mean=%.6f  (combos=%d)\n",
                i, init, s[:avg_mean], s[:n_files])
        if s[:best] !== nothing
            b = s[:best]; w = s[:worst]
            @printf("    best : mean=%.6f, std=%.6f, file=%s\n", b[:mean], b[:std], b[:file])
            @printf("    worst: mean=%.6f, std=%.6f, file=%s\n", w[:mean], w[:std], w[:file])
        end
    end

    println("\n=== Inits ranked by avg mean_error_overall (k=3) ===")
    rank3 = sort(collect(summaries), by = x -> x[2][3][:avg_mean])
    for (i, (init, sdict)) in enumerate(rank3)
        s = sdict[3]
        @printf("%2d) %-28s  avg_mean=%.6f  (combos=%d)\n",
                i, init, s[:avg_mean], s[:n_files])
        if s[:best] !== nothing
            b = s[:best]; w = s[:worst]
            @printf("    best : mean=%.6f, std=%.6f, file=%s\n", b[:mean], b[:std], b[:file])
            @printf("    worst: mean=%.6f, std=%.6f, file=%s\n", w[:mean], w[:std], w[:file])
        end
    end

    return summaries
end


struct InitKStats
    init::String
    k::Int
    avg_mean::Float64
    avg_std::Float64
    means_per_combo::Vector{Float64}
    best_mean::Float64
    best_file::String
    worst_mean::Float64
    worst_file::String
end

function summarize_for_plots(raw)
    out = InitKStats[]
    for init in sort(collect(keys(raw)))
        for k in (2, 3)
            metrics = raw[init][k]  # vector of NamedTuples (file, systems, mean, std)
            if isempty(metrics)
                push!(out, InitKStats(init, k, NaN, NaN, Float64[], NaN, "", NaN, ""))
                continue
            end
            means = getfield.(metrics, :mean)
            stds  = getfield.(metrics, :std)
            avg_mean = mean(means)
            avg_std  = mean(stds)

            bi = argmin(means); wi = argmax(means)
            best = metrics[bi]; worst = metrics[wi]
            push!(out, InitKStats(init, k, avg_mean, avg_std, collect(means),
                                  best.mean, best.file, worst.mean, worst.file))
        end
    end
    return out
end

function get_stat(init, k)
    first(filter(s -> s.init == init && s.k == k, stats))
end

function fig_grouped_bars(inits; ylabel="avg mean_error_overall", title="Average performance by init")
    xs = 1:length(inits)
    means_k2 = [get_stat(init, 2).avg_mean for init in inits]
    means_k3 = [get_stat(init, 3).avg_mean for init in inits]
    errs_k2  = [get_stat(init, 2).avg_std  for init in inits]
    errs_k3  = [get_stat(init, 3).avg_std  for init in inits]

    f = Figure(resolution=(1200, 550))
    ax = Axis(f[1,1], title=title, xticks=(xs, inits), xticklabelrotation=35,
              ylabel=ylabel)
    width = 0.35
    barplot!(ax, xs .- width/2, means_k2, width=width, label="k=2")
    barplot!(ax, xs .+ width/2, means_k3, width=width, label="k=3")
    errorbars!(ax, xs .- width/2, means_k2, errs_k2, whiskerwidth=10)
    errorbars!(ax, xs .+ width/2, means_k3, errs_k3, whiskerwidth=10)
    axislegend(ax)
    hidespines!(ax)
    f
end

function fig_scatter_k2_vs_k3(inits; title="k=2 vs k=3 (avg mean per init)")
    x = [get_stat(init, 2).avg_mean for init in inits]
    y = [get_stat(init, 3).avg_mean for init in inits]

    f = Figure(resolution=(700, 650))
    ax = Axis(f[1,1], xlabel="avg mean (k=2)", ylabel="avg mean (k=3)", title=title)
    lim = (minimum([x;y]) * 0.95, maximum([x;y]) * 1.05)
    scatter!(ax, x, y)
    for (xi, yi, name) in zip(x, y, inits)
        text!(ax, string(name), position=(xi, yi), align=(:left, :bottom), fontsize=10)
    end
    hidespines!(ax)
    f
end

# Beeswarm/strip plot: Makie has no native violin, and beeswarm + median line
# reads cleanly for a journal figure.
function fig_distributions(stats; title="Distribution across system combinations")
    f = Figure(resolution=(1400, 650))
    for (col, k) in enumerate((2,3))
        subset = filter(s -> s.k == k && !isempty(s.means_per_combo), stats)
        inits_k = sort(unique(s.init for s in subset))
        ax = Axis(f[1,col], title="k=$(k)", xticks=(1:length(inits_k), inits_k),
                  xticklabelrotation=35, ylabel="mean_error_overall")
        for (i, init) in enumerate(inits_k)
            s = first(filter(t -> t.init == init && t.k == k, subset))
            n = length(s.means_per_combo)
            jitter = (rand(n) .- 0.5) .* 0.6
            xs = fill(i, n) .+ jitter
            scatter!(ax, xs, s.means_per_combo, markersize=5, transparency=true)
            med = median(s.means_per_combo)
            lines!(ax, [i-0.35, i+0.35], [med, med], linewidth=3)
        end
        hidespines!(ax)
    end
    f
end



function print_best_worst(stats)
    println("\n=== Best & Worst per init and k ===")
    for init in sort(unique(s.init for s in stats))
        println("\n[$init]")
        for k in (2,3)
            s = get_stat(init, k)
            if isempty(s.means_per_combo)
                println("  (k=$k) no data")
                continue
            end
            @printf("  (k=%d) best : mean=%.6f  file=%s\n", k, s.best_mean,  s.best_file)
            @printf("         worst: mean=%.6f  file=%s\n", s.worst_mean, s.worst_file)
        end
    end
end

ordered_reservoir_inits = [
    "cycle_jumps",
    "delay_line",
    "double_cycle",
    "selfloop_delayline_backward",
    "forward_connection",
    "selfloop_cycle",
    "selfloop_forwardconnection",
    "simple_cycle",
    "selfloop_backward_cycle",
    "delayline_backward"
]

bt_root = "bt_results"

# Only keep inits that actually exist as directories
reservoir_inits = filter(init -> isdir(joinpath(bt_root, init)),
                         ordered_reservoir_inits)

results_by_init = Dict{String, Any}()

for init in reservoir_inits
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    errors = Float64[]
    stds   = Float64[]

    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f), allownan=true)
        push!(errors, obj["mean_error_overall"])
        push!(stds,   obj["std_error_overall"])
    end
    errors = filter(!isnan, errors)
    stds = filter(!isnan, stds)

    med      = median(errors)
    q25      = quantile(errors, 0.25)
    q75      = quantile(errors, 0.75)

    med_std  = median(stds)
    q25_std  = quantile(stds, 0.25)
    q75_std  = quantile(stds, 0.75)

    results_by_init[init] = Dict(
        "errors" => errors,
        "median" => med,
        # error bars stored as distances from median
        "lower_error" => med - q25,
        "upper_error" => q75 - med,

        "stds" => stds,
        "median_std" => med_std,
        "lower_error_std" => med_std - q25_std,
        "upper_error_std" => q75_std - med_std
    )
end

mkpath("results")
open("results/results_by_init_bt.json", "w") do io
    JSON.print(io, results_by_init)
end

parc_root = "parc_results"

# Only keep inits that actually exist as directories
reservoir_inits = filter(init -> isdir(joinpath(parc_root, init)),
                         ordered_reservoir_inits)

results_by_init = Dict{String, Any}()

for init in reservoir_inits
    init_dir = joinpath(parc_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    errors = Float64[]
    stds   = Float64[]

    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f), allownan=true)
        push!(errors, obj["mean_error_overall"])
        push!(stds,   obj["std_error_overall"])
    end
    errors = filter(!isnan, errors)
    stds = filter(!isnan, stds)

    med      = median(errors)
    q25      = quantile(errors, 0.25)
    q75      = quantile(errors, 0.75)

    med_std  = median(stds)
    q25_std  = quantile(stds, 0.25)
    q75_std  = quantile(stds, 0.75)

    results_by_init[init] = Dict(
        "errors" => errors,
        "median" => med,
        # error bars stored as distances from median
        "lower_error" => med - q25,
        "upper_error" => q75 - med,

        "stds" => stds,
        "median_std" => med_std,
        "lower_error_std" => med_std - q25_std,
        "upper_error_std" => q75_std - med_std
    )
end

mkpath("results")
open("results/results_by_init_parc.json", "w") do io
    JSON.print(io, results_by_init)
end

SYSTEMS = [ChaoticDynamicalSystemLibrary.Aizawa,
    ChaoticDynamicalSystemLibrary.Chen,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Halvorsen,
    ChaoticDynamicalSystemLibrary.Lorenz,
    ChaoticDynamicalSystemLibrary.Rossler,
    ChaoticDynamicalSystemLibrary.SprottS]

known_names = string.(SYSTEMS)
summaries_split = summarize_parc_results_split(known_names, "bt_results")
raw = collect_metrics_by_k("parc_results", known_names)
stats = summarize_for_plots(raw)

inits = sort(unique(s.init for s in stats))

fA = fig_grouped_bars(inits)
save("panel_A_grouped_bars.pdf", fA)

fB = fig_scatter_k2_vs_k3(inits)
save("panel_B_scatter_k2_vs_k3.pdf", fB)

fC = fig_distributions(stats)
save("panel_C_distributions.pdf", fC)

print_best_worst(summaries_split)

println("\nSaved figures:")
println("  panel_A_grouped_bars.pdf")
println("  panel_B_scatter_k2_vs_k3.pdf")
println("  panel_C_distributions.pdf")
