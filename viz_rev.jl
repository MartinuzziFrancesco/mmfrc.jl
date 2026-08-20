using CairoMakie
using JSON
using Random
using Colors
using Statistics
using ChaoticDynamicalSystemLibrary, OrdinaryDiffEq, OrdinaryDiffEqFeagin, StatsBase

Random.seed!(42)

include("base/theme.jl")
include("base/data.jl")
include("base/evaluation.jl")


systems = [
    ChaoticDynamicalSystemLibrary.Aizawa,
    ChaoticDynamicalSystemLibrary.Arneodo,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Halvorsen,
    ChaoticDynamicalSystemLibrary.Lorenz,
    ChaoticDynamicalSystemLibrary.Rossler,
    ChaoticDynamicalSystemLibrary.SprottS,
]

chaotic_data = solve_csystems(systems)


parc_root = "parc_results"

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
    "delayline_backward",
]

init_labels = Dict(
    "cycle_jumps" => "CJ",
    "delay_line" => "DL",
    "double_cycle" => "DC",
    "selfloop_delayline_backward" => "SLDB",
    "forward_connection" => "FC",
    "selfloop_cycle" => "SLC",
    "selfloop_forwardconnection" => "SLFC",
    "simple_cycle" => "SC",
    "selfloop_backward_cycle" => "SLFB",
    "delayline_backward" => "DLFB",
)

label_to_init = Dict(v => k for (k, v) in init_labels)


function blend_colors(c1::RGB, c2::RGB, t::Float64)
    RGB(
        (1 - t) * c1.r + t * c2.r,
        (1 - t) * c1.g + t * c2.g,
        (1 - t) * c1.b + t * c2.b,
    )
end

function forecast_to_matrix(raw_forecast, expected_rows::Int)
    tmp = hcat(raw_forecast...)
    if size(tmp, 1) == expected_rows
        return tmp
    elseif size(tmp, 2) == expected_rows
        return permutedims(tmp)
    else
        error("Could not reshape forecast to $(expected_rows)×T. Got size $(size(tmp)).")
    end
end

function collect_metric_by_init(root::String, inits::Vector{String}, key::String)
    out = Dict{String, Vector{Float64}}()

    for init in inits
        init_dir = joinpath(root, init)
        vals = Float64[]

        if !isdir(init_dir)
            @warn "Missing directory: $init_dir"
            out[init] = vals
            continue
        end

        files = filter(f -> endswith(f, ".json"), readdir(init_dir))

        for f in files
            fpath = joinpath(init_dir, f)
            obj = JSON.parsefile(fpath, allownan=true)

            if haskey(obj, key)
                append!(vals, Float64.(obj[key]))
            else
                @warn "Missing key $key in $fpath"
            end
        end

        out[init] = vals
    end

    return out
end

function plot_metric_violins!(
    ax,
    metric_by_init::Dict{String, Vector{Float64}},
    reversed_inits::Vector{String},
    color_map::Dict{String, RGB};
    logscale=true,
    eps_val=1e-12,
    markersize=28,
)
    for (i, init) in enumerate(reversed_inits)
        raw_vals = metric_by_init[init]

        if isempty(raw_vals)
            @warn "No values for $init"
            continue
        end

        vals = logscale ? log10.(max.(raw_vals, eps_val)) : raw_vals
        med = median(vals)
        base = color_map[init]

        violin!(
            ax,
            fill(i, length(vals)),
            vals;
            width = 0.8,
            color = RGBA(base, 0.6),
            strokecolor = base,
            strokewidth = 1.5,
            show_median = false,
        )

        scatter!(
            ax,
            [i],
            [med];
            color = :black,
            markersize = markersize,
        )
    end

    return nothing
end

function find_parc_result(root::String, init::String, systems_pair::Tuple{String,String})
    init_dir = joinpath(root, init)
    isdir(init_dir) || error("Missing init directory: $init_dir")

    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    for f in files
        fpath = joinpath(init_dir, f)
        obj = JSON.parsefile(fpath, allownan=true)
        systems = Tuple(String.(obj["systems"]))
        if systems == systems_pair
            return fpath, obj
        end
    end

    error("Could not find PARC result for init=$init and systems=$(systems_pair)")
end

function get_parc_reconstruction(
    root::String,
    init::String,
    systems_pair::Tuple{String,String},
    run_idx::Int,
    chaotic_data,
)
    fpath, obj = find_parc_result(root, init, systems_pair)

    meta = obj["meta"]
    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift0 = 50
    shift = shift0 + (run_idx - 1) * 50

    data_pair, _ = build_labelled_attractors_separated(
        train_len,
        predict_len,
        chaotic_data;
        system_names = collect(systems_pair),
        shift = shift,
    )

    truth1 = data_pair[3][1]
    truth2 = data_pair[3][2]

    raw_forecast = obj["forecasts"][run_idx]
    pred1 = forecast_to_matrix(raw_forecast[1], size(truth1, 1))
    pred2 = forecast_to_matrix(raw_forecast[2], size(truth2, 1))

    kld = haskey(obj, "per_run_error_overall") ?
        Float64(obj["per_run_error_overall"][run_idx]) : NaN

    tvar = haskey(obj, "per_run_total_variation_overall") ?
        Float64(obj["per_run_total_variation_overall"][run_idx]) : NaN

    return (
        fpath    = fpath,
        systems  = systems_pair,
        init     = init,
        run_idx  = run_idx,
        kld      = kld,
        tvar     = tvar,
        truth1   = truth1,
        truth2   = truth2,
        pred1    = pred1,
        pred2    = pred2,
    )
end

function plot_parc_pair_3d!(
    ax3d,
    rec;
    N=350,
    pred_start=40,
    linewidth=4.8,
)
    color_palette = Makie.colorschemes[:seaborn_muted]

    sys1_color = color_palette[3]
    sys2_color = color_palette[4]

    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    # keep truth starting at 1, but prediction starts at pred_start
    n1 = min(N, size(rec.truth1, 2), size(rec.pred1, 2) - pred_start + 1)
    n2 = min(N, size(rec.truth2, 2), size(rec.pred2, 2) - pred_start + 1)

    ttruth1 = 1:n1
    ttruth2 = 1:n2
    tpred1  = pred_start:(pred_start + n1 - 1)
    tpred2  = pred_start:(pred_start + n2 - 1)

    lines!(
        ax3d,
        rec.truth1[1, ttruth1],
        rec.truth1[2, ttruth1],
        rec.truth1[3, ttruth1];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred1[1, tpred1],
        rec.pred1[2, tpred1],
        rec.pred1[3, tpred1];
        color = sys1_color,
        linewidth = linewidth,
    )

    lines!(
        ax3d,
        rec.truth2[1, ttruth2],
        rec.truth2[2, ttruth2],
        rec.truth2[3, ttruth2];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred2[1, tpred2],
        rec.pred2[2, tpred2],
        rec.pred2[3, tpred2];
        color = sys2_color,
        linewidth = linewidth,
    )

    cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-8, 8, 0))

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][] = false
    axis3[:showticks][] = false
    axis3[:showgrid][] = false

    return nothing
end

# Hand-picked representative PARC pairs, in display order:
# 1. Lorenz-Rossler  2. GenesioTesi-SprottS  3. Arneodo-SprottS

selected_examples = [
    (
        systems = ("Lorenz", "Rossler"),
        init = label_to_init["FC"],      # forward_connection
        run_idx = 1,
        Nplot = 350,
        pred_start = 40,
    ),
    (
        systems = ("GenesioTesi", "SprottS"),
        init = label_to_init["DLFB"],    # delayline_backward
        run_idx = 10,
        Nplot = 350,
        pred_start = 1,
    ),
    (
        systems = ("Arneodo", "Rossler"),
        init = label_to_init["FC"],      # delay_line
        run_idx = 7,
        Nplot = 350,
        pred_start = 1, 
    ),
]

selected_recs = [
    get_parc_reconstruction(
        parc_root,
        ex.init,
        ex.systems,
        ex.run_idx,
        chaotic_data,
    )
    for ex in selected_examples
]

println("Selected PARC examples:")
for (i, rec) in enumerate(selected_recs)
    println(
        "[$i] $(rec.systems[1])-$(rec.systems[2]) | " *
        "init=$(rec.init) | run=$(rec.run_idx) | " *
        "log10(KLD)=$(round(log10(max(rec.kld, 1e-12)); digits=3)) | " *
        "TVar=$(round(rec.tvar; digits=3))"
    )
end

# PARC metric distributions

kld_tmp = collect_metric_by_init(
    parc_root,
    ordered_reservoir_inits,
    "per_run_error_overall",
)

inits = filter(init -> haskey(kld_tmp, init) && !isempty(kld_tmp[init]), ordered_reservoir_inits)

ordered_by_parc = sort(
    inits;
    by = init -> median(log10.(max.(kld_tmp[init], 1e-12))),
    rev = true,
)

reversed_inits = reverse(ordered_by_parc)
reversed_labels = [init_labels[init] for init in reversed_inits]

kld_parc = collect_metric_by_init(
    parc_root,
    reversed_inits,
    "per_run_error_overall",
)

tvar_parc = collect_metric_by_init(
    parc_root,
    reversed_inits,
    "per_run_total_variation_overall",
)


color_palette = Makie.colorschemes[:seaborn_muted]
base_color = RGB(color_palette[1])
light_target = RGB(0.95, 0.95, 0.95)

color_map = Dict{String, RGB}()
shade_params = range(0.0, 0.8, length=length(reversed_inits))

for (i, init) in enumerate(reversed_inits)
    color_map[init] = blend_colors(base_color, light_target, shade_params[i])
end


fig = Figure(resolution = (1800, 950))

maingrid   = fig[1, 1] = GridLayout()
topgrid    = maingrid[1, 1] = GridLayout()
bottomgrid = maingrid[2, 1] = GridLayout()

# top row: 3 chosen PARC attractor examples

for j in 1:3
    rec = selected_recs[j]
    Nplot = selected_examples[j].Nplot

    ax3d = LScene(
        topgrid[1, j],
        scenekw = (show_axis = false,),
    )

    plot_parc_pair_3d!(
        ax3d,
        rec;
        N = Nplot,
        pred_start = selected_examples[j].pred_start,
        linewidth = 4.8,
    )
end

# bottom row: PARC metric distributions

ax_kld = Axis(
    bottomgrid[1, 1],
    ylabel = "log₁₀ KLD",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 28,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 2.5, 5.0, 7.5],
)

plot_metric_violins!(
    ax_kld,
    kld_parc,
    reversed_inits,
    color_map;
    logscale = true,
)

ylims!(ax_kld, -2.5, 10)

ax_tvar = Axis(
    bottomgrid[1, 2],
    ylabel = "TVar",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 28,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 0.5, 1.0],
)

plot_metric_violins!(
    ax_tvar,
    tvar_parc,
    reversed_inits,
    color_map;
    logscale = false,
)

ylims!(ax_tvar, -0.05, 1.10)


Label(
    topgrid[1, 1, TopLeft()],
    "(a)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    topgrid[1, 2, TopLeft()],
    "(b)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    topgrid[1, 3, TopLeft()],
    "(c)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    bottomgrid[1, 1, TopLeft()],
    "(d)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    bottomgrid[1, 2, TopLeft()],
    "(e)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)


rowsize!(maingrid, 1, Relative(0.58))
rowsize!(maingrid, 2, Relative(0.42))

colgap!(topgrid, 15)
colgap!(bottomgrid, 35)
rowgap!(maingrid, 10)

fig

mkpath("figures")
save("figures/fig04new.eps", fig, dpi=600)
save("figures/fig04new.png", fig, dpi=600)




parc_root = "parc_results"

init_labels = Dict(
    "cycle_jumps" => "CJ",
    "delay_line" => "DL",
    "double_cycle" => "DC",
    "selfloop_delayline_backward" => "SLDB",
    "forward_connection" => "FC",
    "selfloop_cycle" => "SLC",
    "selfloop_forwardconnection" => "SLFC",
    "simple_cycle" => "SC",
    "selfloop_backward_cycle" => "SLFB",
    "delayline_backward" => "DLFB",
)


function forecast_to_matrix(raw_forecast, expected_rows::Int)
    tmp = hcat(raw_forecast...)
    if size(tmp, 1) == expected_rows
        return tmp
    elseif size(tmp, 2) == expected_rows
        return permutedims(tmp)
    else
        error("Could not reshape forecast to $(expected_rows)×T. Got size $(size(tmp)).")
    end
end

function all_parc_result_files(root::String)
    files = String[]

    for init in readdir(root)
        init_dir = joinpath(root, init)
        isdir(init_dir) || continue

        for f in readdir(init_dir)
            endswith(f, ".json") || continue
            push!(files, joinpath(init_dir, f))
        end
    end

    return sort(files)
end

"""
Select the best individual PARC runs across all files.

metric_key can be:
- "per_run_error_overall"            KLD-based selection
- "per_run_total_variation_overall"  TVar-based selection
"""
function best_parc_runs(root::String; nbest=9, metric_key="per_run_error_overall")
    files = all_parc_result_files(root)

    rows = NamedTuple[]

    for fpath in files
        obj = JSON.parsefile(fpath, allownan=true)
        init = splitpath(fpath)[end - 1]
        systems = String.(obj["systems"])

        haskey(obj, metric_key) || continue
        scores = Float64.(obj[metric_key])

        for run_idx in eachindex(scores)
            push!(rows, (
                fpath   = fpath,
                init    = init,
                systems = systems,
                run_idx = run_idx,
                score   = scores[run_idx],
                obj     = obj,
            ))
        end
    end

    rows = sort(rows; by = r -> r.score)
    return rows[1:min(nbest, length(rows))]
end

function get_parc_reconstruction(row, chaotic_data)
    obj = row.obj
    run_idx = row.run_idx

    meta = obj["meta"]
    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift0 = 50
    shift = shift0 + (run_idx - 1) * 50

    data_pair, _ = build_labelled_attractors_separated(
        train_len,
        predict_len,
        chaotic_data;
        system_names = row.systems,
        shift = shift,
    )

    truth1 = data_pair[3][1]
    truth2 = data_pair[3][2]

    raw_forecast = obj["forecasts"][run_idx]

    pred1 = forecast_to_matrix(raw_forecast[1], size(truth1, 1))
    pred2 = forecast_to_matrix(raw_forecast[2], size(truth2, 1))

    kld = haskey(obj, "per_run_error_overall") ?
        Float64(obj["per_run_error_overall"][run_idx]) : NaN

    tvar = haskey(obj, "per_run_total_variation_overall") ?
        Float64(obj["per_run_total_variation_overall"][run_idx]) : NaN

    return (
        systems = row.systems,
        init    = row.init,
        run_idx = run_idx,
        kld     = kld,
        tvar    = tvar,
        truth1  = truth1,
        truth2  = truth2,
        pred1   = pred1,
        pred2   = pred2,
    )
end

function plot_parc_pair_3d!(
    ax3d,
    rec;
    N=350,
    linewidth=4.2,
)
    color_palette = Makie.colorschemes[:seaborn_muted]

    sys1_color = color_palette[3]
    sys2_color = color_palette[4]

    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    n1 = min(N, size(rec.truth1, 2), size(rec.pred1, 2))
    n2 = min(N, size(rec.truth2, 2), size(rec.pred2, 2))

    lines!(
        ax3d,
        rec.truth1[1, 1:n1],
        rec.truth1[2, 1:n1],
        rec.truth1[3, 1:n1];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred1[1, 1:n1],
        rec.pred1[2, 1:n1],
        rec.pred1[3, 1:n1];
        color = sys1_color,
        linewidth = linewidth,
    )

    lines!(
        ax3d,
        rec.truth2[1, 1:n2],
        rec.truth2[2, 1:n2],
        rec.truth2[3, 1:n2];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred2[1, 1:n2],
        rec.pred2[2, 1:n2],
        rec.pred2[3, 1:n2];
        color = sys2_color,
        linewidth = linewidth,
    )

    cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-8, 8, 0))

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][]  = false
    axis3[:showticks][] = false
    axis3[:showgrid][]  = false

    return nothing
end

function slice_best_parc_runs(root::String; start_idx=1, nshow=9, metric_key="per_run_error_overall")
    rows = best_parc_runs(root; nbest=start_idx + nshow - 1, metric_key=metric_key)
    return rows[start_idx:start_idx + nshow - 1]
end

# Pick best 9 PARC runs

best_rows = slice_best_parc_runs(
    parc_root;
    start_idx = 1250,
    nshow = 9,
    metric_key = "per_run_total_variation_overall",
)

# If instead you want best 9 by TVar, use:
# best_rows = slice_best_parc_runs(
#     parc_root;
#     start_idx = 1,
#     nshow = 9,
#     metric_key = "per_run_total_variation_overall",
# )

best_recs = [get_parc_reconstruction(row, chaotic_data) for row in best_rows]

println("Selected 9 individual PARC runs:")
for (i, rec) in enumerate(best_recs)
    println(
        "[$i] $(rec.systems[1])-$(rec.systems[2]) | " *
        "init=$(rec.init) | run=$(rec.run_idx) | " *
        "log10(KLD)=$(round(log10(max(rec.kld, 1e-12)); digits=3)) | " *
        "TVar=$(round(rec.tvar; digits=3))"
    )
end


fig = Figure(resolution = (1800, 1600))
grid = fig[1, 1] = GridLayout()

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

for idx in 1:length(best_recs)
    rec = best_recs[idx]

    r = cld(idx, 3)
    c = idx - 3 * (r - 1)

    cell = grid[r, c] = GridLayout()

    ax3d = LScene(
        cell[1, 1],
        scenekw = (show_axis = false,),
    )

    plot_parc_pair_3d!(
        ax3d,
        rec;
        N = 350,
        linewidth = 4.2,
    )

    Label(
        cell[1, 1, TopLeft()],
        panel_labels[idx];
        fontsize = 22,
        font = :bold,
        padding = (8, 8, 8, 8),
        halign = :left,
    )

    Label(
        cell[2, 1],
        "$(rec.systems[1])--$(rec.systems[2])\n" *
        "$(init_labels[rec.init]), run $(rec.run_idx)\n" *
        "log₁₀KLD = $(round(log10(max(rec.kld, 1e-12)); digits=2)), " *
        "TVar = $(round(rec.tvar; digits=2))";
        fontsize = 18,
        halign = :center,
        tellheight = true,
    )
end

colgap!(grid, 20)
rowgap!(grid, 20)

fig

mkpath("figures")
save("figures/fig_parc_selected9_examples.eps", fig, dpi=600)
save("figures/fig_parc_selected9_examples.png", fig, dpi=600)
























bt_root = "bt_results"

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
    "delayline_backward",
]

init_labels = Dict(
    "cycle_jumps" => "CJ",
    "delay_line" => "DL",
    "double_cycle" => "DC",
    "selfloop_delayline_backward" => "SLDB",
    "forward_connection" => "FC",
    "selfloop_cycle" => "SLC",
    "selfloop_forwardconnection" => "SLFC",
    "simple_cycle" => "SC",
    "selfloop_backward_cycle" => "SLFB",
    "delayline_backward" => "DLFB",
)

label_to_init = Dict(v => k for (k, v) in init_labels)


function blend_colors(c1::RGB, c2::RGB, t::Float64)
    RGB(
        (1 - t) * c1.r + t * c2.r,
        (1 - t) * c1.g + t * c2.g,
        (1 - t) * c1.b + t * c2.b,
    )
end

function forecast_to_matrix(raw_forecast, expected_rows::Int)
    tmp = hcat(raw_forecast...)
    if size(tmp, 1) == expected_rows
        return tmp
    elseif size(tmp, 2) == expected_rows
        return permutedims(tmp)
    else
        error("Could not reshape forecast to $(expected_rows)×T. Got size $(size(tmp)).")
    end
end

function collect_metric_by_init(root::String, inits::Vector{String}, key::String)
    out = Dict{String, Vector{Float64}}()

    for init in inits
        init_dir = joinpath(root, init)
        vals = Float64[]

        if !isdir(init_dir)
            @warn "Missing directory: $init_dir"
            out[init] = vals
            continue
        end

        files = filter(f -> endswith(f, ".json"), readdir(init_dir))

        for f in files
            fpath = joinpath(init_dir, f)
            obj = JSON.parsefile(fpath, allownan=true)

            if haskey(obj, key)
                append!(vals, Float64.(obj[key]))
            else
                @warn "Missing key $key in $fpath"
            end
        end

        out[init] = vals
    end

    return out
end

function plot_metric_violins!(
    ax,
    metric_by_init::Dict{String, Vector{Float64}},
    reversed_inits::Vector{String},
    color_map::Dict{String, RGB};
    logscale=true,
    eps_val=1e-12,
    markersize=28,
)
    for (i, init) in enumerate(reversed_inits)
        raw_vals = metric_by_init[init]

        if isempty(raw_vals)
            @warn "No values for $init"
            continue
        end

        vals = logscale ? log10.(max.(raw_vals, eps_val)) : raw_vals
        med = median(vals)
        base = color_map[init]

        violin!(
            ax,
            fill(i, length(vals)),
            vals;
            width = 0.8,
            color = RGBA(base, 0.6),
            strokecolor = base,
            strokewidth = 1.5,
            show_median = false,
        )

        scatter!(
            ax,
            [i],
            [med];
            color = :black,
            markersize = markersize,
        )
    end

    return nothing
end

function find_bt_result(root::String, init::String, systems_pair::Tuple{String,String})
    init_dir = joinpath(root, init)
    isdir(init_dir) || error("Missing init directory: $init_dir")

    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    for f in files
        fpath = joinpath(init_dir, f)
        obj = JSON.parsefile(fpath, allownan=true)
        systems = Tuple(String.(obj["systems"]))
        if systems == systems_pair
            return fpath, obj
        end
    end

    error("Could not find BT result for init=$init and systems=$(systems_pair)")
end

function get_bt_reconstruction(
    root::String,
    init::String,
    systems_pair::Tuple{String,String},
    run_idx::Int,
    chaotic_data,
)
    fpath, obj = find_bt_result(root, init, systems_pair)

    meta = obj["meta"]
    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift0 = 50
    shift = shift0 + (run_idx - 1) * 50

    data_pair, _ = build_pair_separated(
        train_len,
        predict_len,
        chaotic_data;
        system_names = collect(systems_pair),
        shift = shift,
    )

    truth = data_pair[3]  # 6 × T

    raw_forecast = obj["forecasts"][run_idx]
    forecast = forecast_to_matrix(raw_forecast, 6)

    kld = haskey(obj, "per_run_error_overall") ?
        Float64(obj["per_run_error_overall"][run_idx]) : NaN

    tvar = haskey(obj, "per_run_total_variation_overall") ?
        Float64(obj["per_run_total_variation_overall"][run_idx]) : NaN

    return (
        fpath    = fpath,
        systems  = systems_pair,
        init     = init,
        run_idx  = run_idx,
        kld      = kld,
        tvar     = tvar,
        truth    = truth,
        forecast = forecast,
    )
end

function plot_bt_pair_3d!(
    ax3d,
    rec;
    N=350,
    linewidth=4.8,
)
    color_palette = Makie.colorschemes[:seaborn_muted]

    sys1_color = color_palette[3]
    sys2_color = color_palette[4]

    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    n = min(N, size(rec.truth, 2), size(rec.forecast, 2))

    lines!(
        ax3d,
        rec.truth[1, 1:n],
        rec.truth[2, 1:n],
        rec.truth[3, 1:n];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[1, 1:n],
        rec.forecast[2, 1:n],
        rec.forecast[3, 1:n];
        color = sys1_color,
        linewidth = linewidth,
    )

    lines!(
        ax3d,
        rec.truth[4, 1:n],
        rec.truth[5, 1:n],
        rec.truth[6, 1:n];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[4, 1:n],
        rec.forecast[5, 1:n],
        rec.forecast[6, 1:n];
        color = sys2_color,
        linewidth = linewidth,
    )

    cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-8, 8, 0))

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][] = false
    axis3[:showticks][] = false
    axis3[:showgrid][] = false

    return nothing
end

# Hand-picked representative BT pairs, in display order:
# 1. Chua-Lorenz  2. Chua-SprottS  3. Halvorsen-Rossler

selected_examples = [
    (
        systems = ("Chua", "Lorenz"),
        init = label_to_init["DC"],       # double_cycle
        run_idx = 6,
        Nplot = 350,
    ),
    (
        systems = ("Chua", "SprottS"),
        init = label_to_init["SLFC"],     # selfloop_forwardconnection
        run_idx = 9,
        Nplot = 350,
    ),
    (
        systems = ("Halvorsen", "Rossler"),
        init = label_to_init["SLFC"],
        run_idx = 9,
        Nplot = 350,
    ),
]

selected_recs = [
    get_bt_reconstruction(
        bt_root,
        ex.init,
        ex.systems,
        ex.run_idx,
        chaotic_data,
    )
    for ex in selected_examples
]

println("Selected BT examples:")
for (i, rec) in enumerate(selected_recs)
    println(
        "[$i] $(rec.systems[1])-$(rec.systems[2]) | " *
        "init=$(rec.init) | run=$(rec.run_idx) | " *
        "log10(KLD)=$(round(log10(max(rec.kld, 1e-12)); digits=3)) | " *
        "TVar=$(round(rec.tvar; digits=3))"
    )
end

# BT metric distributions

kld_tmp = collect_metric_by_init(
    bt_root,
    ordered_reservoir_inits,
    "per_run_error_overall",
)

inits = filter(init -> haskey(kld_tmp, init) && !isempty(kld_tmp[init]), ordered_reservoir_inits)

ordered_by_bt = sort(
    inits;
    by = init -> median(log10.(max.(kld_tmp[init], 1e-12))),
    rev = true,
)

reversed_inits = reverse(ordered_by_bt)
reversed_labels = [init_labels[init] for init in reversed_inits]

kld_bt = collect_metric_by_init(
    bt_root,
    reversed_inits,
    "per_run_error_overall",
)

tvar_bt = collect_metric_by_init(
    bt_root,
    reversed_inits,
    "per_run_total_variation_overall",
)


color_palette = Makie.colorschemes[:seaborn_muted]
base_color = RGB(color_palette[1])
light_target = RGB(0.95, 0.95, 0.95)

color_map = Dict{String, RGB}()
shade_params = range(0.0, 0.8, length=length(reversed_inits))

for (i, init) in enumerate(reversed_inits)
    color_map[init] = blend_colors(base_color, light_target, shade_params[i])
end


fig = Figure(resolution = (1800, 950))

maingrid   = fig[1, 1] = GridLayout()
topgrid    = maingrid[1, 1] = GridLayout()
bottomgrid = maingrid[2, 1] = GridLayout()

# top row: 3 chosen BT attractor examples

for j in 1:3
    rec = selected_recs[j]
    Nplot = selected_examples[j].Nplot

    ax3d = LScene(
        topgrid[1, j],
        scenekw = (show_axis = false,),
    )

    plot_bt_pair_3d!(
        ax3d,
        rec;
        N = Nplot,
        linewidth = 4.8,
    )
end

# bottom row: BT metric distributions

ax_kld = Axis(
    bottomgrid[1, 1],
    ylabel = "log₁₀ KLD",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 28,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 2.5, 5.0, 7.5],
)

plot_metric_violins!(
    ax_kld,
    kld_bt,
    reversed_inits,
    color_map;
    logscale = true,
)

ylims!(ax_kld, -2.5, 10)

ax_tvar = Axis(
    bottomgrid[1, 2],
    ylabel = "TVar",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 28,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 0.5, 1.0],
)

plot_metric_violins!(
    ax_tvar,
    tvar_bt,
    reversed_inits,
    color_map;
    logscale = false,
)

ylims!(ax_tvar, -0.05, 1.10)


Label(
    topgrid[1, 1, TopLeft()],
    "(a)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    topgrid[1, 2, TopLeft()],
    "(b)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    topgrid[1, 3, TopLeft()],
    "(c)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    bottomgrid[1, 1, TopLeft()],
    "(d)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    bottomgrid[1, 2, TopLeft()],
    "(e)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)


rowsize!(maingrid, 1, Relative(0.58))
rowsize!(maingrid, 2, Relative(0.42))

colgap!(topgrid, 15)
colgap!(bottomgrid, 35)
rowgap!(maingrid, 10)

fig

mkpath("figures")
save("figures/fig05new.eps", fig, dpi=600)
save("figures/fig05new.png", fig, dpi=600)





bt_root = "bt_results"

init_labels = Dict(
    "cycle_jumps" => "CJ",
    "delay_line" => "DL",
    "double_cycle" => "DC",
    "selfloop_delayline_backward" => "SLDB",
    "forward_connection" => "FC",
    "selfloop_cycle" => "SLC",
    "selfloop_forwardconnection" => "SLFC",
    "simple_cycle" => "SC",
    "selfloop_backward_cycle" => "SLFB",
    "delayline_backward" => "DLFB",
)


function forecast_to_matrix(raw_forecast, expected_rows::Int)
    tmp = hcat(raw_forecast...)
    if size(tmp, 1) == expected_rows
        return tmp
    elseif size(tmp, 2) == expected_rows
        return permutedims(tmp)
    else
        error("Could not reshape forecast to $(expected_rows)×T. Got size $(size(tmp)).")
    end
end

function all_bt_result_files(root::String)
    files = String[]

    for init in readdir(root)
        init_dir = joinpath(root, init)
        isdir(init_dir) || continue

        for f in readdir(init_dir)
            endswith(f, ".json") || continue
            push!(files, joinpath(init_dir, f))
        end
    end

    return sort(files)
end

"""
Select the best individual BT/ICT runs across all files.

metric_key can be:
- "per_run_error_overall"            KLD-based selection
- "per_run_total_variation_overall"  TVar-based selection
"""
function best_bt_runs(root::String; nbest=9, metric_key="per_run_error_overall")
    files = all_bt_result_files(root)

    rows = NamedTuple[]

    for fpath in files
        obj = JSON.parsefile(fpath, allownan=true)
        init = splitpath(fpath)[end - 1]
        systems = String.(obj["systems"])

        haskey(obj, metric_key) || continue
        scores = Float64.(obj[metric_key])

        for run_idx in eachindex(scores)
            push!(rows, (
                fpath   = fpath,
                init    = init,
                systems = systems,
                run_idx = run_idx,
                score   = scores[run_idx],
                obj     = obj,
            ))
        end
    end

    rows = sort(rows; by = r -> r.score)
    return rows[1:min(nbest, length(rows))]
end

function get_bt_reconstruction(row, chaotic_data)
    obj = row.obj
    run_idx = row.run_idx

    meta = obj["meta"]
    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift0 = 50
    shift = shift0 + (run_idx - 1) * 50

    data_pair, _ = build_pair_separated(
        train_len,
        predict_len,
        chaotic_data;
        system_names = row.systems,
        shift = shift,
    )

    truth = data_pair[3]  # 6 × T

    raw_forecast = obj["forecasts"][run_idx]
    forecast = forecast_to_matrix(raw_forecast, 6)

    kld = haskey(obj, "per_run_error_overall") ?
        Float64(obj["per_run_error_overall"][run_idx]) : NaN

    tvar = haskey(obj, "per_run_total_variation_overall") ?
        Float64(obj["per_run_total_variation_overall"][run_idx]) : NaN

    return (
        systems = row.systems,
        init    = row.init,
        run_idx = run_idx,
        kld     = kld,
        tvar    = tvar,
        truth   = truth,
        forecast = forecast,
    )
end

function plot_bt_pair_3d!(
    ax3d,
    rec;
    N=350,
    linewidth=4.2,
)
    color_palette = Makie.colorschemes[:seaborn_muted]

    sys1_color = color_palette[3]
    sys2_color = color_palette[4]

    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    n = min(N, size(rec.truth, 2), size(rec.forecast, 2))

    # system 1: rows 1:3
    lines!(
        ax3d,
        rec.truth[1, 1:n],
        rec.truth[2, 1:n],
        rec.truth[3, 1:n];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[1, 1:n],
        rec.forecast[2, 1:n],
        rec.forecast[3, 1:n];
        color = sys1_color,
        linewidth = linewidth,
    )

    # system 2: rows 4:6
    lines!(
        ax3d,
        rec.truth[4, 1:n],
        rec.truth[5, 1:n],
        rec.truth[6, 1:n];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[4, 1:n],
        rec.forecast[5, 1:n],
        rec.forecast[6, 1:n];
        color = sys2_color,
        linewidth = linewidth,
    )

    cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-8, 8, 0))

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][]  = false
    axis3[:showticks][] = false
    axis3[:showgrid][]  = false

    return nothing
end

function slice_best_bt_runs(root::String; start_idx=1, nshow=9, metric_key="per_run_error_overall")
    rows = best_bt_runs(root; nbest=start_idx + nshow - 1, metric_key=metric_key)
    return rows[start_idx:start_idx + nshow - 1]
end

# Pick best 9 BT/ICT runs

# Best 9 by KLD
best_rows = slice_best_bt_runs(
    bt_root;
    start_idx = 50,
    nshow = 9,
    metric_key = "per_run_error_overall",
)
# If instead you want best 9 by TVar, use:
# best_rows = best_bt_runs(bt_root; nbest=9, metric_key="per_run_total_variation_overall")

best_recs = [get_bt_reconstruction(row, chaotic_data) for row in best_rows]

println("Best 9 individual BT/ICT runs:")
for (i, rec) in enumerate(best_recs)
    println(
        "[$i] $(rec.systems[1])-$(rec.systems[2]) | " *
        "init=$(rec.init) | run=$(rec.run_idx) | " *
        "log10(KLD)=$(round(log10(max(rec.kld, 1e-12)); digits=3)) | " *
        "TVar=$(round(rec.tvar; digits=3))"
    )
end


fig = Figure(resolution = (1800, 1600))
grid = fig[1, 1] = GridLayout()

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

for idx in 1:length(best_recs)
    rec = best_recs[idx]

    r = cld(idx, 3)
    c = idx - 3 * (r - 1)

    cell = grid[r, c] = GridLayout()

    ax3d = LScene(
        cell[1, 1],
        scenekw = (show_axis = false,),
    )

    plot_bt_pair_3d!(
        ax3d,
        rec;
        N = 350,
        linewidth = 4.2,
    )

    Label(
        cell[1, 1, TopLeft()],
        panel_labels[idx];
        fontsize = 22,
        font = :bold,
        padding = (8, 8, 8, 8),
        halign = :left,
    )

    Label(
        cell[2, 1],
        "$(rec.systems[1])--$(rec.systems[2])\n" *
        "$(init_labels[rec.init]), run $(rec.run_idx)\n" *
        "log₁₀KLD = $(round(log10(max(rec.kld, 1e-12)); digits=2)), " *
        "TVar = $(round(rec.tvar; digits=2))";
        fontsize = 18,
        halign = :center,
        tellheight = true,
    )
end

colgap!(grid, 20)
rowgap!(grid, 20)

fig

mkpath("figures")

save("bt_9best.png", fig, dpi=600)


