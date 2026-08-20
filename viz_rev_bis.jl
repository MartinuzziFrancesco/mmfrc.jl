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


ict_root  = "bt_results"
parc_root = "parc_results"
obt_root = "obt_results"

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

mkpath("figures")


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

function order_inits_by_metric(root::String, inits::Vector{String}; key="per_run_error_overall")
    metric = collect_metric_by_init(root, inits, key)
    present_inits = filter(init -> haskey(metric, init) && !isempty(metric[init]), inits)

    ordered = sort(
        present_inits;
        by = init -> median(log10.(max.(metric[init], 1e-12))),
        rev = true,
    )

    return reverse(ordered)
end

function build_color_map(reversed_inits)
    color_palette = Makie.colorschemes[:seaborn_muted]
    base_color = RGB(color_palette[1])
    light_target = RGB(0.95, 0.95, 0.95)

    color_map = Dict{String, RGB}()
    shade_params = range(0.0, 0.8, length=length(reversed_inits))

    for (i, init) in enumerate(reversed_inits)
        color_map[init] = blend_colors(base_color, light_target, shade_params[i])
    end

    return color_map
end

function plot_metric_violins!(
    ax,
    metric_by_init::Dict{String, Vector{Float64}},
    reversed_inits::Vector{String},
    color_map::Dict{String, RGB};
    logscale=true,
    eps_val=1e-12,
    markersize=30,
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

function fit_camera_to_data!(ax3d, xs, ys, zs; dist_factor = 1.45f0)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    zmin, zmax = extrema(zs)

    center = Vec3f(
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2,
    )

    spanx = xmax - xmin
    spany = ymax - ymin
    spanz = zmax - zmin
    span = max(spanx, spany, spanz)

    span == 0 && (span = 1f0)

    eyepos = center + Vec3f(-dist_factor * span, dist_factor * span, 0.35f0 * span)
    cam3d!(ax3d; lookat = center, eyeposition = eyepos)

    return nothing
end

# Result-file lookup helpers

function find_result(root::String, init::String, systems_pair::Tuple{String,String})
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

    error("Could not find result for init=$init and systems=$(systems_pair)")
end

# ICT reconstruction helpers

function get_ict_reconstruction(
    root::String,
    init::String,
    systems_pair::Tuple{String,String},
    run_idx::Int,
    chaotic_data,
)
    fpath, obj = find_result(root, init, systems_pair)

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
        fpath = fpath,
        systems = systems_pair,
        init = init,
        run_idx = run_idx,
        kld = kld,
        tvar = tvar,
        truth = truth,
        forecast = forecast,
    )
end

function plot_ict_pair_3d!(
    ax3d,
    rec;
    N=350,
    pred_start=1,
    linewidth=4.8,
    sys1_color = RGB(0.30, 0.45, 0.85),   # blue
    sys2_color = RGB(0.90, 0.55, 0.20),   # orange
)
    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    n = min(N, size(rec.truth, 2), size(rec.forecast, 2) - pred_start + 1)

    truth_idx = 1:n
    pred_idx  = pred_start:(pred_start + n - 1)

    lines!(
        ax3d,
        rec.truth[1, truth_idx],
        rec.truth[2, truth_idx],
        rec.truth[3, truth_idx];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[1, pred_idx],
        rec.forecast[2, pred_idx],
        rec.forecast[3, pred_idx];
        color = sys1_color,
        linewidth = linewidth,
    )

    lines!(
        ax3d,
        rec.truth[4, truth_idx],
        rec.truth[5, truth_idx],
        rec.truth[6, truth_idx];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.forecast[4, pred_idx],
        rec.forecast[5, pred_idx],
        rec.forecast[6, pred_idx];
        color = sys2_color,
        linewidth = linewidth,
    )

    xs = vcat(
    vec(rec.truth[1, truth_idx]), vec(rec.forecast[1, pred_idx]),
    vec(rec.truth[4, truth_idx]), vec(rec.forecast[4, pred_idx]),
)
ys = vcat(
    vec(rec.truth[2, truth_idx]), vec(rec.forecast[2, pred_idx]),
    vec(rec.truth[5, truth_idx]), vec(rec.forecast[5, pred_idx]),
)
zs = vcat(
    vec(rec.truth[3, truth_idx]), vec(rec.forecast[3, pred_idx]),
    vec(rec.truth[6, truth_idx]), vec(rec.forecast[6, pred_idx]),
)

fit_camera_to_data!(ax3d, xs, ys, zs; dist_factor = 1.9f0)

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][] = false
    axis3[:showticks][] = false
    axis3[:showgrid][] = false

    return nothing
end

# PARC reconstruction helpers

function get_parc_reconstruction(
    root::String,
    init::String,
    systems_pair::Tuple{String,String},
    run_idx::Int,
    chaotic_data,
)
    fpath, obj = find_result(root, init, systems_pair)

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
        fpath = fpath,
        systems = systems_pair,
        init = init,
        run_idx = run_idx,
        kld = kld,
        tvar = tvar,
        truth1 = truth1,
        truth2 = truth2,
        pred1 = pred1,
        pred2 = pred2,
    )
end

function plot_parc_pair_3d!(
    ax3d,
    rec;
    N=350,
    pred_start=1,
    linewidth=4.8,
    sys1_color = RGB(0.35, 0.78, 0.35),   # green
    sys2_color = RGB(0.88, 0.38, 0.38),   # red
)
    truth1_color = RGBA(sys1_color, 0.42)
    truth2_color = RGBA(sys2_color, 0.42)

    n1 = min(N, size(rec.truth1, 2), size(rec.pred1, 2) - pred_start + 1)
    n2 = min(N, size(rec.truth2, 2), size(rec.pred2, 2) - pred_start + 1)

    truth_idx1 = 1:n1
    truth_idx2 = 1:n2

    pred_idx1 = pred_start:(pred_start + n1 - 1)
    pred_idx2 = pred_start:(pred_start + n2 - 1)

    lines!(
        ax3d,
        rec.truth1[1, truth_idx1],
        rec.truth1[2, truth_idx1],
        rec.truth1[3, truth_idx1];
        color = truth1_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred1[1, pred_idx1],
        rec.pred1[2, pred_idx1],
        rec.pred1[3, pred_idx1];
        color = sys1_color,
        linewidth = linewidth,
    )

    lines!(
        ax3d,
        rec.truth2[1, truth_idx2],
        rec.truth2[2, truth_idx2],
        rec.truth2[3, truth_idx2];
        color = truth2_color,
        linewidth = linewidth,
        linestyle = :dash,
    )

    lines!(
        ax3d,
        rec.pred2[1, pred_idx2],
        rec.pred2[2, pred_idx2],
        rec.pred2[3, pred_idx2];
        color = sys2_color,
        linewidth = linewidth,
    )

    xs = vcat(
    vec(rec.truth1[1, truth_idx1]), vec(rec.pred1[1, pred_idx1]),
    vec(rec.truth2[1, truth_idx2]), vec(rec.pred2[1, pred_idx2]),
)
ys = vcat(
    vec(rec.truth1[2, truth_idx1]), vec(rec.pred1[2, pred_idx1]),
    vec(rec.truth2[2, truth_idx2]), vec(rec.pred2[2, pred_idx2]),
)
zs = vcat(
    vec(rec.truth1[3, truth_idx1]), vec(rec.pred1[3, pred_idx1]),
    vec(rec.truth2[3, truth_idx2]), vec(rec.pred2[3, pred_idx2]),
)

fit_camera_to_data!(ax3d, xs, ys, zs; dist_factor = 1.9f0)

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][] = false
    axis3[:showticks][] = false
    axis3[:showgrid][] = false

    return nothing
end

# Metric figure generator

function make_metric_figure(root::String, method_label::String, outfile_prefix::String)
    reversed_inits = order_inits_by_metric(
        root,
        ordered_reservoir_inits;
        key = "per_run_error_overall",
    )

    reversed_labels = [init_labels[init] for init in reversed_inits]
    color_map = build_color_map(reversed_inits)

    kld = collect_metric_by_init(
        root,
        reversed_inits,
        "per_run_error_overall",
    )

    tvar = collect_metric_by_init(
        root,
        reversed_inits,
        "per_run_total_variation_overall",
    )

    fig = Figure(resolution = (760, 950))
    grid = fig[1, 1] = GridLayout()
Label(
    fig[0, 1],
    method_label;
    fontsize = 28,
    font = :bold,
    halign = :center,
    tellwidth = false,
    tellheight = true,
)

    # Top panel: KLD

    ax_kld = Axis(
        grid[1, 1],
        ylabel = "log₁₀ KLD",
        xticks = (1:length(reversed_labels), reversed_labels),
        xticklabelsvisible = false,
        xticksvisible = true,
        ylabelsize = 28,
        xticklabelsize = 22,
        yticklabelsize = 22,
        yticks = [0.0, 2.5, 5.0, 7.5],
    )

    plot_metric_violins!(
        ax_kld,
        kld,
        reversed_inits,
        color_map;
        logscale = true,
    )

    ylims!(ax_kld, -2.5, 10)

    Label(
        grid[1, 1, TopLeft()],
        "(a)";
        fontsize = 24,
        font = :bold,
        padding = (10, 10, 10, 10),
        halign = :left,
    )

    # Bottom panel: TVar

    ax_tvar = Axis(
        grid[2, 1],
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
        tvar,
        reversed_inits,
        color_map;
        logscale = false,
    )

    ylims!(ax_tvar, -0.05, 1.10)

    Label(
        grid[2, 1, TopLeft()],
        "(b)";
        fontsize = 24,
        font = :bold,
        padding = (10, 10, 10, 10),
        halign = :left,
    )


    rowsize!(grid, 1, Relative(0.5))
    rowsize!(grid, 2, Relative(0.5))
    rowgap!(grid, 10)

    save("figures/$(outfile_prefix).eps", fig, dpi=600)
    save("figures/$(outfile_prefix).png", fig, dpi=600)

    return fig
end

# Qualitative example definitions

ict_examples = [
    (
        systems = ("Chua", "Lorenz"),
        init = label_to_init["DC"],
        run_idx = 6,
        Nplot = 350,
        pred_start = 1,
    ),
    (
        systems = ("Chua", "SprottS"),
        init = label_to_init["SLFC"],
        run_idx = 9,
        Nplot = 350,
        pred_start = 1,
    ),
    (
        systems = ("Halvorsen", "Rossler"),
        init = label_to_init["SLFC"],
        run_idx = 9,
        Nplot = 85,
        pred_start = 1,
    ),
]

parc_examples = [
    (
        systems = ("Lorenz", "Rossler"),
        init = label_to_init["FC"],
        run_idx = 1,
        Nplot = 260,
        pred_start = 25,
    ),
    (
        systems = ("GenesioTesi", "SprottS"),
        init = label_to_init["DLFB"],
        run_idx = 10,
        Nplot = 350,
        pred_start = 1,
    ),
    (
        systems = ("Arneodo", "Rossler"),
        init = label_to_init["FC"],
        run_idx = 7,
        Nplot = 350,
        pred_start = 1,
    ),
]

# Qualitative figure generator

function make_qualitative_figure()
    parc_recs = [
        get_parc_reconstruction(
            parc_root,
            ex.init,
            ex.systems,
            ex.run_idx,
            chaotic_data,
        )
        for ex in parc_examples
    ]

    ict_recs = [
        get_ict_reconstruction(
            ict_root,
            ex.init,
            ex.systems,
            ex.run_idx,
            chaotic_data,
        )
        for ex in ict_examples
    ]
    parc_sys1_color = color_palette[3]
    parc_sys2_color = color_palette[7]

    ict_sys1_color = color_palette[3]
    ict_sys2_color = color_palette[7]

    fig = Figure(resolution = (1950, 1040), figure_padding = (0, -100, -100, 0))
    grid = fig[1, 1] = GridLayout()

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # PARC row
    for j in 1:3
        ax3d = LScene(
            grid[1, j],
            scenekw = (show_axis = false,),
        )

        plot_parc_pair_3d!(
            ax3d,
            parc_recs[j];
            N = parc_examples[j].Nplot,
            pred_start = parc_examples[j].pred_start,
            linewidth = 5.2,
            sys1_color = parc_sys1_color,
            sys2_color = parc_sys2_color,
        )

    end

    # ICT row
    for j in 1:3
        ax3d = LScene(
            grid[2, j],
            scenekw = (show_axis = false,),
        )

        plot_ict_pair_3d!(
            ax3d,
            ict_recs[j];
            N = ict_examples[j].Nplot,
            pred_start = ict_examples[j].pred_start,
            linewidth = 5.2,
            sys1_color = ict_sys1_color,
            sys2_color = ict_sys2_color,
        )


    end

    # Row labels
    Label(
        grid[1, 0],
        "PARC";
        fontsize = 35,
        font = :bold,
        rotation = π / 2,
        halign = :center,
        valign = :center,
    )

    Label(
        grid[2, 0],
        "ICT";
        fontsize = 35,
        font = :bold,
        rotation = π / 2,
        halign = :center,
        valign = :center,
    )

text!(fig.scene, "(a)"; position = Point2f(92, 956), fontsize = 30, font = :bold)
text!(fig.scene, "(b)"; position = Point2f(748, 956), fontsize = 30, font = :bold)
text!(fig.scene, "(c)"; position = Point2f(1405, 956), fontsize = 30, font = :bold)

text!(fig.scene, "(d)"; position = Point2f(92, 425), fontsize = 30, font = :bold)
text!(fig.scene, "(e)"; position = Point2f(748, 425), fontsize = 30, font = :bold)
text!(fig.scene, "(f)"; position = Point2f(1405, 425), fontsize = 30, font = :bold)

colsize!(grid, 0, Fixed(34))

rowsize!(grid, 1, Relative(0.5))
rowsize!(grid, 2, Relative(0.5))

# keep the label-to-panel gap modest
colgap!(grid, 1, Fixed(-25))

# aggressively shrink the gaps between the three plot columns
colgap!(grid, 2, Fixed(-300))   # between col 1 and 2
colgap!(grid, 3, Fixed(-300))   # between col 2 and 3

# row spacing
rowgap!(grid, Fixed(-90))

    save("figures/fig_qualitative_parc_ict.eps", fig, dpi=600)
    save("figures/fig_qualitative_parc_ict.png", fig, dpi=600)

    return fig
end

fig_qualitative = make_qualitative_figure()


fig_ict_metrics = make_metric_figure(
    ict_root,
    "Input concatenation training (ICT)", "fig_ict_metrics",
)

fig_parc_metrics = make_metric_figure(
    parc_root,
    "Parameter-aware reservoir computing (PARC)", "fig_parc_metrics",
)

fig_obt_metrics = make_metric_figure(
    obt_root,
    "Blending technique (BT)", "fig_obt_metrics",
)

fig_qualitative = make_qualitative_figure()

fig_ict_metrics
fig_parc_metrics
fig_qualitative

# Additional printout: per-panel summary of the qualitative figure above.

using JSON
using Statistics
using Printf


parc_root = "parc_results"
ict_root  = "bt_results"   # ICT results currently stored here

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

# Selected panels from qualitative figure

examples = [
    (
        panel = "(a)",
        method = "PARC",
        root = parc_root,
        systems = ("Lorenz", "Rossler"),
        init = label_to_init["FC"],
        run_idx = 1,
    ),
    (
        panel = "(b)",
        method = "PARC",
        root = parc_root,
        systems = ("GenesioTesi", "SprottS"),
        init = label_to_init["DLFB"],
        run_idx = 10,
    ),
    (
        panel = "(c)",
        method = "PARC",
        root = parc_root,
        systems = ("Arneodo", "Rossler"),
        init = label_to_init["FC"],
        run_idx = 7,
    ),
    (
        panel = "(d)",
        method = "ICT",
        root = ict_root,
        systems = ("Chua", "Lorenz"),
        init = label_to_init["DC"],
        run_idx = 6,
    ),
    (
        panel = "(e)",
        method = "ICT",
        root = ict_root,
        systems = ("Chua", "SprottS"),
        init = label_to_init["SLFC"],
        run_idx = 9,
    ),
    (
        panel = "(f)",
        method = "ICT",
        root = ict_root,
        systems = ("Halvorsen", "Rossler"),
        init = label_to_init["SLFC"],
        run_idx = 9,
    ),
]


function find_result(root::String, init::String, systems_pair::Tuple{String,String})
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

    error("Could not find result for init=$init and systems=$(systems_pair)")
end

function safe_vec(obj, key::String)
    haskey(obj, key) || return nothing
    return Float64.(obj[key])
end

function mean_log10(vals; eps=1e-12)
    return mean(log10.(max.(vals, eps)))
end

function selected_log10(vals, idx; eps=1e-12)
    return log10(max(vals[idx], eps))
end

function fmt(x)
    if isnan(x)
        return "NaN"
    else
        return @sprintf("%.3f", x)
    end
end

function get_metric_summary(obj, run_idx::Int)
    kld_overall = safe_vec(obj, "per_run_error_overall")
    kld_sys1    = safe_vec(obj, "per_run_error_sys1")
    kld_sys2    = safe_vec(obj, "per_run_error_sys2")

    tvar_overall = safe_vec(obj, "per_run_total_variation_overall")
    tvar_sys1    = safe_vec(obj, "per_run_total_variation_sys1")
    tvar_sys2    = safe_vec(obj, "per_run_total_variation_sys2")

    return (
        selected_log10_kld_overall = isnothing(kld_overall) ? NaN : selected_log10(kld_overall, run_idx),
        selected_tvar_overall      = isnothing(tvar_overall) ? NaN : tvar_overall[run_idx],

        mean_log10_kld_overall = isnothing(kld_overall) ? NaN : mean_log10(kld_overall),
        mean_log10_kld_sys1    = isnothing(kld_sys1) ? NaN : mean_log10(kld_sys1),
        mean_log10_kld_sys2    = isnothing(kld_sys2) ? NaN : mean_log10(kld_sys2),

        mean_tvar_overall = isnothing(tvar_overall) ? NaN : mean(tvar_overall),
        mean_tvar_sys1    = isnothing(tvar_sys1) ? NaN : mean(tvar_sys1),
        mean_tvar_sys2    = isnothing(tvar_sys2) ? NaN : mean(tvar_sys2),

        n_runs = isnothing(kld_overall) ? missing : length(kld_overall),
    )
end


println("\n============================================================")
println("Selected qualitative panels: average metrics by pair/system")
println("============================================================\n")

for ex in examples
    fpath, obj = find_result(ex.root, ex.init, ex.systems)
    s = get_metric_summary(obj, ex.run_idx)

    sys1, sys2 = ex.systems
    init_short = init_labels[ex.init]

    println("------------------------------------------------------------")
    println("Panel:       ", ex.panel)
    println("Method:      ", ex.method)
    println("Systems:     ", sys1, " -- ", sys2)
    println("Initializer: ", ex.init, " (", init_short, ")")
    println("Run index:   ", ex.run_idx)
    println("JSON file:   ", fpath)
    println("n runs:      ", s.n_runs)

    println("\nSelected run:")
    println("  log10(KLD) overall = ", fmt(s.selected_log10_kld_overall))
    println("  TVar overall       = ", fmt(s.selected_tvar_overall))

    println("\nAverage over runs, pair-level:")
    println("  mean log10(KLD) overall = ", fmt(s.mean_log10_kld_overall))
    println("  mean TVar overall       = ", fmt(s.mean_tvar_overall))

    println("\nAverage over runs, individual systems:")
    println("  ", sys1)
    println("    mean log10(KLD) = ", fmt(s.mean_log10_kld_sys1))
    println("    mean TVar       = ", fmt(s.mean_tvar_sys1))

    println("  ", sys2)
    println("    mean log10(KLD) = ", fmt(s.mean_log10_kld_sys2))
    println("    mean TVar       = ", fmt(s.mean_tvar_sys2))
end

println("------------------------------------------------------------")
println("[Done]")