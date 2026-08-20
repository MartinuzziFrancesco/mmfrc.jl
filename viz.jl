using CairoMakie
using JSON
using Random
using Colors
using Statistics
using ChaoticDynamicalSystemLibrary, OrdinaryDiffEq, OrdinaryDiffEqFeagin, FractalDimensions, StatsBase

Random.seed!(42)

include("base/theme.jl")
include("base/data.jl")
include("base/evaluation.jl")

systems = [ChaoticDynamicalSystemLibrary.Aizawa,
    ChaoticDynamicalSystemLibrary.Arneodo,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Halvorsen,
    ChaoticDynamicalSystemLibrary.Lorenz,
    ChaoticDynamicalSystemLibrary.Rossler,
    ChaoticDynamicalSystemLibrary.SprottS]

const csystems = ["Aizawa",
    "Arneodo",
    "Chua",
    "GenesioTesi",
    "Halvorsen",
    "Lorenz",
    "Rossler",
    "SprottS"]

chaotic_data = solve_csystems(systems)

function blend_colors(c1::RGB, c2::RGB, t::Float64)
    RGB(
        (1 - t) * c1.r + t * c2.r,
        (1 - t) * c1.g + t * c2.g,
        (1 - t) * c1.b + t * c2.b,
    )
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
    scale_factor=1.0,
)
    for (i, init) in enumerate(reversed_inits)
        raw_vals = metric_by_init[init]

        if isempty(raw_vals)
            @warn "No values for $init"
            continue
        end

        scaled_vals = scale_factor .* raw_vals
        vals = logscale ? log10.(max.(scaled_vals, eps_val)) : scaled_vals
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

function set_separated_view!(ax, truth; rows1=1:3, rows2=4:6, dist_factor=3f0)
    c1 = Vec3f(
        mean(truth[rows1[1], :]),
        mean(truth[rows1[2], :]),
        mean(truth[rows1[3], :]),
    )
    c2 = Vec3f(
        mean(truth[rows2[1], :]),
        mean(truth[rows2[2], :]),
        mean(truth[rows2[3], :]),
    )

    mid  = (c1 + c2) / 2
    line = c2 - c1

    ref = abs(dot(line, Vec3f(0, 0, 1))) < 0.9 ? Vec3f(0, 0, 1) : Vec3f(0, 1, 0)

    dir = normalize(cross(line, ref))
    dist = dist_factor * norm(line)

    eyepos = mid + dist * dir

    cam3d!(ax; lookat = mid, eyeposition = eyepos)
    return nothing
end

results_bt = JSON.parsefile("results/results_by_init_bt.json")
results_parc = JSON.parsefile("results/results_by_init_parc.json")

ordered_reservoir_inits = sort(
    [
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
    ];
    by = init -> results_bt[init]["median"],
    rev = true,
)

inits = filter(init -> haskey(results_bt, init), ordered_reservoir_inits)
reversed_inits = reverse(inits)

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

reversed_labels = [init_labels[init] for init in reversed_inits]

color_palette = Makie.colorschemes[:seaborn_muted]
base_color = RGB(color_palette[1])
light_target = RGB(0.95, 0.95, 0.95)

color_map = Dict{String, RGB}()
shade_params = range(0.0, 0.8, length=length(reversed_inits))

for (i, init) in enumerate(reversed_inits)
    color_map[init] = blend_colors(base_color, light_target, shade_params[i])
end

bt_root = "bt_results"
parc_root = "parc_results"

kld_bt = collect_metric_by_init(
    bt_root,
    reversed_inits,
    "per_run_error_overall",
)

kld_parc = collect_metric_by_init(
    parc_root,
    reversed_inits,
    "per_run_error_overall",
)

tvar_bt = collect_metric_by_init(
    bt_root,
    reversed_inits,
    "per_run_total_variation_overall",
)

tvar_parc = collect_metric_by_init(
    parc_root,
    reversed_inits,
    "per_run_total_variation_overall",
)

fig = Figure(resolution = (1800, 800))

mainf = fig[1, 1] = GridLayout()
leftfig = mainf[1, 1] = GridLayout()
rightfig = mainf[1, 2] = GridLayout()

kld_bt_fig    = leftfig[1, 1] = GridLayout()
adev_bt_fig   = leftfig[1, 2] = GridLayout()
kld_parc_fig  = leftfig[2, 1] = GridLayout()
adev_parc_fig = leftfig[2, 2] = GridLayout()

ax_kld_bt = Axis(
    kld_bt_fig[1, 1],
    ylabel = rich(rich("BT", font = :bold), "\n\n", "log₁₀ KLD"),
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelsvisible = false,
    ylabelsize = 26,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 2.5, 5.0, 7.5],
)

plot_metric_violins!(
    ax_kld_bt,
    kld_bt,
    reversed_inits,
    color_map;
    logscale = true,
)

ylims!(ax_kld_bt, -2.5, 10)

ax_adev_bt = Axis(
    adev_bt_fig[1, 1],
    ylabel = "TVar",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelsvisible = false,
    ylabelsize = 26,
    xticklabelsize = 22,
    yticklabelsize = 22,
)

plot_metric_violins!(
    ax_adev_bt,
    tvar_bt,
    reversed_inits,
    color_map;
    logscale = false,
)

ax_kld_parc = Axis(
    kld_parc_fig[1, 1],
    ylabel = rich(rich("PARC", font = :bold), "\n\n", "log₁₀ KLD"),
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 26,
    xticklabelsize = 22,
    yticklabelsize = 22,
    yticks = [0.0, 2.5, 5.0, 7.5],
)

plot_metric_violins!(
    ax_kld_parc,
    kld_parc,
    reversed_inits,
    color_map;
    logscale = true,
)

ylims!(ax_kld_parc, -2.5, 10)

ax_adev_parc = Axis(
    adev_parc_fig[1, 1],
    ylabel = "TVar",
    xticks = (1:length(reversed_labels), reversed_labels),
    xticklabelrotation = π / 4,
    ylabelsize = 26,
    xticklabelsize = 22,
    yticklabelsize = 22,
)

plot_metric_violins!(
    ax_adev_parc,
    tvar_parc,
    reversed_inits,
    color_map;
    logscale = false,
)

fig
init = "cycle_jumps"
init_dir = joinpath(bt_root, init)

files = filter(f -> endswith(f, ".json"), readdir(init_dir))

pairs = [
    let obj = JSON.parsefile(joinpath(init_dir, f), allownan=true)
        (file = f, mean = obj["mean_error_overall"], obj = obj)
    end
    for f in files
]

sorted_pairs = sort(pairs; by = p -> p.mean)

length(sorted_pairs) ≥ 3 || error("Need at least three system pairs for $init")

second = sorted_pairs[3]  # third-best pair by mean error
second_obj = second.obj
best_systems = String.(second_obj["systems"])

errs = second_obj["per_run_error_overall"]
order = sortperm(errs)
med_idx = order[cld(length(order), 2)]
run_idx = med_idx

meta = second_obj["meta"]
train_len = meta["train_len"]
predict_len = meta["predict_len"]

println("Representative pair for $(init):")
println("  systems = $best_systems")
println("  run idx = $run_idx")
println("  KLD/CDE = ", errs[run_idx])

if haskey(second_obj, "per_run_attractor_deviation_overall")
    println("  TVar    = ", second_obj["per_run_attractor_deviation_overall"][run_idx])
end

shift0 = 50
shift = shift0 + (run_idx - 1) * 50

data_pair, _ = build_pair_separated(
    Int(train_len),
    Int(predict_len),
    chaotic_data;
    system_names = best_systems,
    shift = shift,
)

truth = data_pair[3]

raw_forecast = second_obj["forecasts"][run_idx]
tmp = hcat(raw_forecast...)
forecast = size(tmp, 1) == 6 ? tmp : permutedims(tmp)

tvar_ymin = -0.05
tvar_ymax = 1.15
tvar_ticks = [0.0, 0.5, 1.0]

ylims!(ax_adev_bt, tvar_ymin, tvar_ymax)
ylims!(ax_adev_parc, tvar_ymin, tvar_ymax)

ax_adev_bt.yticks = tvar_ticks
ax_adev_parc.yticks = tvar_ticks

ax3d = LScene(rightfig[1, 1], scenekw = (show_axis = false,))

sys1_color = color_palette[3]
sys2_color = color_palette[4]

truth1_color = RGBA(sys1_color, 0.45)
truth2_color = RGBA(sys2_color, 0.45)

line_width = 5.3
N = 300

lines!(
    ax3d,
    truth[1, 1:N],
    truth[2, 1:N],
    truth[3, 1:N];
    color = truth1_color,
    linewidth = line_width,
    linestyle = :dash,
)

lines!(
    ax3d,
    forecast[1, 1:N],
    forecast[2, 1:N],
    forecast[3, 1:N];
    color = sys1_color,
    linewidth = line_width,
)

lines!(
    ax3d,
    truth[4, 1:N],
    truth[5, 1:N],
    truth[6, 1:N];
    color = truth2_color,
    linewidth = line_width,
    linestyle = :dash,
)

lines!(
    ax3d,
    forecast[4, 1:N],
    forecast[5, 1:N],
    forecast[6, 1:N];
    color = sys2_color,
    linewidth = line_width,
)

cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-6, 6, 0))

axis3 = ax3d.scene.plots[1]
axis3[:showaxis][] = false
axis3[:showticks][] = false
axis3[:showgrid][] = false

Label(
    leftfig[1, 1, TopLeft()],
    "(a)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    leftfig[1, 2, TopLeft()],
    "(b)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    leftfig[2, 1, TopLeft()],
    "(c)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    leftfig[2, 2, TopLeft()],
    "(d)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

Label(
    rightfig[1, 1, TopLeft()],
    "(e)";
    fontsize = 24,
    font = :bold,
    padding = (10, 10, 10, 10),
    halign = :left,
)

colsize!(mainf, 1, Relative(0.68))
colsize!(mainf, 2, Relative(0.32))

colgap!(leftfig, 25)
rowgap!(leftfig, 20)

fig

mkpath("figures")
save("figures/fig04.eps", fig, dpi=600)
save("figures/fig04.png", fig, dpi=600)

# Additional diagnostics printed to stdout (bimodal-split summary of the
# per-topology log10 KLD distributions); not part of the saved figure.

using Statistics
using StatsBase
using KernelDensity
using Printf

function summarize_logkld(raw_errors; eps=1e-12)
    errs = max.(raw_errors, eps)
    loge = log10.(errs)
    return (
        n      = length(loge),
        mean   = mean(loge),
        median = median(loge),
        q25    = quantile(loge, 0.25),
        q75    = quantile(loge, 0.75),
        min    = minimum(loge),
        max    = maximum(loge),
    )
end

# Detect two dominant modes via KDE, then split at the valley between them.
# Returns nothing if it can't confidently find two peaks.
function bimodal_split_kde(loge; npoints=512)
    kd = kde(loge; npoints=npoints)
    xs, ys = kd.x, kd.density

    maxima = Int[]
    for i in 2:length(ys)-1
        if ys[i] > ys[i-1] && ys[i] > ys[i+1]
            push!(maxima, i)
        end
    end
    length(maxima) < 2 && return nothing

    maxima_sorted = sort(maxima; by=i->ys[i], rev=true)
    p1, p2 = maxima_sorted[1], maxima_sorted[2]
    p1, p2 = min(p1, p2), max(p1, p2)

    # valley (minimum density) between the two peak positions
    if p2 - p1 < 3
        return nothing  # peaks too close to make a meaningful split
    end
    rng = p1:p2
    valley_i = rng[argmin(@view ys[rng])]
    thresh = xs[valley_i]

    g1 = loge[loge .<= thresh]
    g2 = loge[loge .>  thresh]
    (length(g1) == 0 || length(g2) == 0) && return nothing

    return (
        threshold = thresh,
        mean1     = mean(g1),
        mean2     = mean(g2),
        median1   = median(g1),
        median2   = median(g2),
        n1        = length(g1),
        n2        = length(g2),
        p1        = length(g1) / length(loge),
        p2        = length(g2) / length(loge),
    )
end

# Set results_dict to the results container to summarize (BT or PARC).
results_dict = results_parc
inits_in_plot_order = reversed_inits

summ = Dict{String, Any}()

for init in inits_in_plot_order
    raw_errors = results_dict[init]["errors"]
    s = summarize_logkld(raw_errors)
    loge = log10.(max.(raw_errors, 1e-12))
    b = bimodal_split_kde(loge)
    summ[init] = (stats=s, bimodal=b)
end

best_init = argmin(init -> summ[init].stats.median, inits_in_plot_order)
best_stats = summ[best_init].stats

@printf("\nBest topology (by median log10KLD): %s\n", best_init)
@printf("  median log10KLD = %.4f  (KLD ≈ %.4e)\n", best_stats.median, 10.0^(best_stats.median))
@printf("  mean   log10KLD = %.4f  (KLD ≈ %.4e)\n", best_stats.mean,   10.0^(best_stats.mean))
@printf("  IQR    log10KLD = [%.4f, %.4f]\n\n", best_stats.q25, best_stats.q75)

for init in inits_in_plot_order
    s = summ[init].stats
    @printf("%-18s  median=%.3f  mean=%.3f  IQR=[%.3f, %.3f]\n",
            String(init), s.median, s.mean, s.q25, s.q75)

    b = summ[init].bimodal
    if b !== nothing
        @printf("  bimodal split @ %.3f: μ1=%.3f (p=%.2f, n=%d), μ2=%.3f (p=%.2f, n=%d)\n",
                b.threshold, b.mean1, b.p1, b.n1, b.mean2, b.p2, b.n2)
    end
end

best_init = argmin(init -> summ[init].stats.median, inits_in_plot_order)
bs = summ[best_init].stats

@printf("\nBest topology (lowest median log10KLD): %s\n", String(best_init))
@printf("  median log10KLD = %.4f   =>  median KLD ≈ %.4e\n", bs.median, 10.0^bs.median)
@printf("  mean   log10KLD = %.4f   =>  mean   KLD ≈ %.4e\n", bs.mean,   10.0^bs.mean)
@printf("  IQR    log10KLD = [%.4f, %.4f]\n\n", bs.q25, bs.q75)

for init in inits_in_plot_order
    b = summ[init].bimodal
    b === nothing && continue

    @printf("%s bimodal lobes:\n", String(init))
    @printf("  split @ log10KLD = %.3f\n", b.threshold)
    @printf("  lobe1 mean log10KLD = %.3f (KLD≈%.3e), p=%.2f, n=%d\n",
            b.mean1, 10.0^b.mean1, b.p1, b.n1)
    @printf("  lobe2 mean log10KLD = %.3f (KLD≈%.3e), p=%.2f, n=%d\n\n",
            b.mean2, 10.0^b.mean2, b.p2, b.n2)
end

μ1s = Float64[]
μ2s = Float64[]
p1s = Float64[]
p2s = Float64[]
ths = Float64[]

for init in inits_in_plot_order
    b = summ[init].bimodal
    b === nothing && continue
    push!(μ1s, b.mean1)
    push!(μ2s, b.mean2)
    push!(p1s, b.p1)
    push!(p2s, b.p2)
    push!(ths, b.threshold)
end

@printf("Across %d topologies with a detected split:\n", length(μ1s))
@printf("  <μ1> = %.3f (std=%.3f)\n", mean(μ1s), std(μ1s))
@printf("  <μ2> = %.3f (std=%.3f)\n", mean(μ2s), std(μ2s))
@printf("  <p1> = %.2f (std=%.2f)\n", mean(p1s), std(p1s))
@printf("  <p2> = %.2f (std=%.2f)\n", mean(p2s), std(p2s))
@printf("  <split threshold> = %.3f (std=%.3f)\n", mean(ths), std(ths))

@printf("\nIn KLD units (via 10^x):\n")
@printf("  10^{<μ1>} ≈ %.3e\n", 10.0^(mean(μ1s)))
@printf("  10^{<μ2>} ≈ %.3e\n", 10.0^(mean(μ2s)))

bt_root   = "bt_results"
parc_root = "parc_results"

inits_all = ordered_reservoir_inits

inits_bt   = filter(init -> isdir(joinpath(bt_root, init)), inits_all)
inits_parc = filter(init -> isdir(joinpath(parc_root, init)), inits_all)

# use intersection so both panels have the same columns, preserving order
inits = [init for init in inits_all if init in inits_bt && init in inits_parc]
n_inits = length(inits)

# use one init as reference to define the pair ordering
ref_init = first(inits)
ref_dir  = joinpath(bt_root, ref_init)

ref_files = sort(filter(f ->
    startswith(f, "fixed_") && endswith(f, ".json"),
    readdir(ref_dir),
))

function system_code(name::AbstractString)
    if name == "Aizawa"
        return "Ai"
    elseif name == "Arneodo"
        return "Ar"
    else
        return string(first(name))  # e.g. "Lorenz" -> "L"
    end
end

pair_labels = String[]
for f in ref_files
    obj = JSON.parsefile(joinpath(ref_dir, f))
    sys = obj["systems"]  # e.g. ["Lorenz", "Rossler"]
    label = string(system_code(sys[1]), "-", system_code(sys[2]))
    push!(pair_labels, label)
end

n_pairs = length(pair_labels)

# builds the pairs x inits heat matrix from a given results root
function build_heat(root::AbstractString, inits, ref_files)
    H = fill(NaN, n_pairs, length(inits))
    for (ix, init) in enumerate(inits)
        init_dir = joinpath(root, init)
        for (iy, f) in enumerate(ref_files)
            path = joinpath(init_dir, f)
            if isfile(path)
                obj = JSON.parsefile(path; allownan=true)
                H[iy, ix] = obj["mean_error_overall"]
            end
        end
    end
    return H
end

heat_bt   = build_heat(bt_root,   inits, ref_files)
heat_parc = build_heat(parc_root, inits, ref_files)

eps_val = 1e-12

safe_bt   = max.(heat_bt,   eps_val)
safe_parc = max.(heat_parc, eps_val)

log_bt   = log10.(safe_bt)
log_parc = log10.(safe_parc)

valid_vals = vcat(log_bt[.!isnan.(log_bt)], log_parc[.!isnan.(log_parc)])
crange_log = (minimum(valid_vals), maximum(valid_vals))

fig = Figure(resolution = (800, 1200))
mainf    = fig[1, 1] = GridLayout()
legendfg = fig[2, 1] = GridLayout()
leftfig  = mainf[1, 1] = GridLayout()
rightfig = mainf[1, 2] = GridLayout()

ax_bt = Axis(
    leftfig[1, 1],
    xlabel = "Reservoir initializer",
    ylabel = "System pair",
    xticks = (1:n_inits, [init_labels[i] for i in inits]),
    yticks = (1:n_pairs, pair_labels),
    xticklabelrotation = π/4,
    ylabelsize = 28,
    xlabelsize = 28,
    xticklabelsize = 20,
    yticklabelsize = 20,
)

hm_bt = heatmap!(ax_bt, log_bt';
    colormap   = :viridis,
    colorrange = crange_log,
)

ax_parc = Axis(
    rightfig[1, 1],
    xlabel = "Reservoir initializer",
    yticklabelsvisible = false,
    yticks = (1:n_pairs, pair_labels),
    xticks = (1:n_inits, [init_labels[i] for i in inits]),
    xticklabelrotation = π/4,
    ylabelsize = 28,
    xlabelsize = 28,
    xticklabelsize = 20,
    yticklabelsize = 20,
)

hm_parc = heatmap!(ax_parc, log_parc';
    colormap   = :viridis,
    colorrange = crange_log,
)

Colorbar(
    legendfg[1, 1], hm_bt;
    vertical   = false,
    height     = 27,
    label      = "log₁₀ KLD",
    labelsize  = 28,
    flipaxis   = false,
    ticksize   = 15,
    tickwidth  = 4,
    spinewidth = 3,
    tickalign  = 0.5,
    ticklabelsize = 24,
)

Label(
    leftfig[1, 1, TopLeft()], "(a)";
    fontsize = 24,
    font     = :bold,
    padding  = (10, 10, 10, 10),
    halign   = :left,
)

Label(
    rightfig[1, 1, TopLeft()], "(b)";
    fontsize = 24,
    font     = :bold,
    padding  = (10, 10, 10, 10),
    halign   = :left,
)

fig

save("figures/fig05.eps", fig, dpi=600)
save("figures/fig05.png", fig, dpi=600)

# Additional diagnostics printed to stdout: per-pair rankings from the heat
# matrices above; not part of the saved figure.

using Statistics
using StatsBase
using Printf

# safe stats on a vector (ignores NaNs)
function vec_stats(v::AbstractVector)
    w = v[.!isnan.(v)]
    isempty(w) && return nothing
    return (
        n      = length(w),
        mean   = mean(w),
        median = median(w),
        std    = std(w),
        q25    = quantile(w, 0.25),
        q75    = quantile(w, 0.75),
        min    = minimum(w),
        max    = maximum(w),
    )
end

# Row-wise summary for a matrix M with shape (n_pairs, n_inits)
# Returns a vector of NamedTuples, one per pair label.
function row_summaries(M, pair_labels, inits, init_labels)
    rows = NamedTuple[]
    for (iy, plab) in enumerate(pair_labels)
        row = @view M[iy, :]
        s = vec_stats(row)
        s === nothing && continue

        valid = .!isnan.(row)
        vals  = row[valid]
        idxs  = findall(valid)  # maps back into original column indices

        best_local = argmin(vals)
        worst_local = argmax(vals)

        best_ix = idxs[best_local]
        worst_ix = idxs[worst_local]

        best_init = inits[best_ix]
        worst_init = inits[worst_ix]

        push!(rows, (
            pair      = plab,
            n         = s.n,
            mean      = s.mean,
            median    = s.median,
            std       = s.std,
            q25       = s.q25,
            q75       = s.q75,
            min       = s.min,
            max       = s.max,
            best_init = init_labels[best_init],
            best_val  = row[best_ix],
            worst_init= init_labels[worst_init],
            worst_val = row[worst_ix],
        ))
    end
    return rows
end

function print_row_ranking(rows; k=5, title="")
    println("\n", title)
    println("Easiest rows (lowest mean log10 mean KLD):")
    for r in sort(rows; by = x -> x.mean)[1:min(k, length(rows))]
        @printf("  %-6s  mean=%.3f  med=%.3f  IQR=[%.3f, %.3f]  best=%s(%.3f)\n",
                r.pair, r.mean, r.median, r.q25, r.q75, r.best_init, r.best_val)
    end
    println("\nHardest rows (highest mean log10 mean KLD):")
    for r in sort(rows; by = x -> x.mean, rev=true)[1:min(k, length(rows))]
        @printf("  %-6s  mean=%.3f  med=%.3f  IQR=[%.3f, %.3f]  worst=%s(%.3f)\n",
                r.pair, r.mean, r.median, r.q25, r.q75, r.worst_init, r.worst_val)
    end
end

rows_bt   = row_summaries(log_bt,   pair_labels, inits, init_labels)
rows_parc = row_summaries(log_parc, pair_labels, inits, init_labels)

print_row_ranking(rows_bt;   k=7, title="BT panel (a): row-wise ranking")
print_row_ranking(rows_parc; k=7, title="PA panel (b): row-wise ranking")

# sorted easiest -> hardest
function print_full_rows(rows; title="")
    println("\n", title)
    for r in sort(rows; by = x -> x.mean)
        @printf("%-6s  mean=%.3f  med=%.3f  std=%.3f  IQR=[%.3f, %.3f]  min=%.3f  max=%.3f  best=%s(%.3f)\n",
                r.pair, r.mean, r.median, r.std, r.q25, r.q75, r.min, r.max, r.best_init, r.best_val)
    end
end

# Uncomment for the full listing:
# print_full_rows(rows_bt;   title="BT: all rows (sorted by mean)")
# print_full_rows(rows_parc; title="PA: all rows (sorted by mean)")

using JSON, Statistics, CairoMakie, GeometryBasics

const SHIFT0 = 50

# 1. Find the best bt and parc runs
using JSON, Statistics

function find_kth_best_pair(root::AbstractString, inits_all, k::Int)
    entries = NamedTuple[]

    for init in inits_all
        init_dir = joinpath(root, init)
        isdir(init_dir) || continue

        files = filter(f -> endswith(f, ".json"), readdir(init_dir))
        for f in files
            path = joinpath(init_dir, f)
            obj  = JSON.parsefile(path; allownan = true)
            errs = obj["per_run_error_overall"]

            local_min, run_idx = findmin(errs)
            push!(entries, (root = root, init = init,
                            obj = obj, run_idx = run_idx,
                            file = f, err = local_min))
        end
    end

    @assert !isempty(entries) "No valid JSONs in $root"
    sorted = sort(entries; by = e -> e.err)
    @assert k ≤ length(sorted) "Requested k=$k but only $(length(sorted)) entries"

    return sorted[k]
end

inits_all = ordered_reservoir_inits

best_bt   = find_kth_best_pair("bt_results",   inits_all, 1)  # best
best_parc = find_kth_best_pair("parc_results", inits_all, 4)

println("Best bt_results run:")
println("  init   = ", best_bt.init)
println("  file   = ", best_bt.file)
println("  CDE    = ", best_bt.err)

println("Second-best parc_results run:")
println("  init   = ", best_parc.init)
println("  file   = ", best_parc.file)
println("  CDE    = ", best_parc.err)

# 2. Rebuild truth + forecast

# BT: forecasts are single 6×T matrices (encoded row-wise)
function rebuild_bt(best)
    obj     = best.obj
    run_idx = best.run_idx

    systems = String.(obj["systems"])
    meta    = obj["meta"]

    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift = SHIFT0 + (run_idx - 1) * 50

    data_pair, _ = build_pair_separated(
        train_len, predict_len, chaotic_data;
        system_names = systems,
        shift = shift,
    )
    truth = data_pair[3]  # 6 × predict_len

    raw_forecast = obj["forecasts"][run_idx]
    tmp = hcat(raw_forecast...)  # 6×T or T×6
    forecast = size(tmp, 1) == 6 ? tmp : permutedims(tmp)

    return (truth = truth, forecast = forecast,
            systems = systems, train_len = train_len,
            predict_len = predict_len)
end

# PAESN: forecasts are a Vector of per-system outputs (3×T each)
function rebuild_parc(best)
    obj     = best.obj
    run_idx = best.run_idx

    systems = String.(obj["systems"])
    meta    = obj["meta"]

    train_len   = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])

    shift = SHIFT0 + (run_idx - 1) * 50

    data_pair, _ = build_pair_separated(
        train_len, predict_len, chaotic_data;
        system_names = systems,
        shift = shift,
    )
    truth = data_pair[3]  # 6 × predict_len

    raw_forecasts = obj["forecasts"][run_idx]  # length 2 for blended systems

    sys_mats = Matrix{Float64}[]
    for rf in raw_forecasts
        tmp = hcat(rf...)  # 3×T or T×3
        mat = size(tmp, 1) == 3 ? tmp : permutedims(tmp)
        push!(sys_mats, mat)
    end

    forecast = vcat(sys_mats...)  # stack system 1 and 2 vertically -> 6×T

    return (truth = truth, forecast = forecast,
            systems = systems, train_len = train_len,
            predict_len = predict_len)
end

bt_data   = rebuild_bt(best_bt)
parc_data = rebuild_parc(best_parc)

# 3. Plot
function plot_pair!(layout, data; title = "")
    truth    = data.truth
    forecast = data.forecast

    ax3d = LScene(layout[1, 1], scenekw = (show_axis = false,))

    sys1_color   = color_palette[3]
    sys2_color   = color_palette[4]

    truth1_color = RGBA(sys1_color, 0.45)
    truth2_color = RGBA(sys2_color, 0.45)

    line_width = 5.3
    N = 600

    lines!(ax3d, truth[1, 1:N], truth[2, 1:N], truth[3, 1:N];
        color = truth1_color, linewidth = line_width, linestyle = :dash)
    lines!(ax3d, forecast[1, 1:N], forecast[2, 1:N], forecast[3, 1:N];
        color = sys1_color, linewidth = line_width)

    lines!(ax3d, truth[4, 1:N], truth[5, 1:N], truth[6, 1:N];
        color = truth2_color, linewidth = line_width, linestyle = :dash)
    lines!(ax3d, forecast[4, 1:N], forecast[5, 1:N], forecast[6, 1:N];
        color = sys2_color, linewidth = line_width)

    cam3d!(ax3d; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-8, 8, 0))

    axis3 = ax3d.scene.plots[1]
    axis3[:showaxis][]  = false
    axis3[:showticks][] = false
    axis3[:showgrid][]  = false
end

fig  = Figure(resolution = (1400, 600))
main = fig[1, 1] = GridLayout()
left = main[1, 1] = GridLayout()
right = main[1, 2] = GridLayout()

plot_pair!(left,
    bt_data;
    title = "bt: $(best_bt.init), $(bt_data.systems[1])-$(bt_data.systems[2])\nCDE = $(round(best_bt.err, digits=3))",
)

plot_pair!(right,
    parc_data;
    title = "parc: $(best_parc.init), $(parc_data.systems[1])-$(parc_data.systems[2])\nCDE = $(round(best_parc.err, digits=3))",
)

fig

n_pairs, n_inits = size(heat_bt)

ranks_bt = fill(0, n_pairs, n_inits)

for iy in 1:n_pairs
    row = heat_bt[iy, :]
    valid = .!isnan.(row)
    if !any(valid)
        continue
    end
    ord = sortperm(row[valid])  # ascending errors
    valid_idxs = findall(valid)
    for (rank, k) in enumerate(ord)
        ranks_bt[iy, valid_idxs[k]] = rank
    end
end

top3_frac_bt = zeros(Float64, n_inits)

for j in 1:n_inits
    rj = ranks_bt[:, j]
    valid = rj .> 0
    nj = count(valid)
    nj == 0 && continue
    top3_frac_bt[j] = count(rj[valid] .<= 3) / nj
end

bottom3_frac_bt = zeros(Float64, n_inits)

for iy in 1:n_pairs
    row_ranks = ranks_bt[iy, :]
    valid = row_ranks .> 0
    if !any(valid)
        continue
    end
    maxr = maximum(row_ranks[valid])  # worst rank for this pair
    thresh = max(maxr - 2, 1)  # last 3 ranks: maxr, maxr-1, maxr-2
    for j in 1:n_inits
        r = row_ranks[j]
        if r > 0
            if r >= thresh
                bottom3_frac_bt[j] += 1
            end
        end
    end
end

# normalize by number of pairs each init actually appears in
for j in 1:n_inits
    rj = ranks_bt[:, j]
    nj = count(rj .> 0)
    nj == 0 && continue
    bottom3_frac_bt[j] /= nj
end

fig = Figure(resolution = (800, 600))
ax  = Axis(fig[1, 1],
    xlabel = "Fraction of pairs in top 3",
    ylabel = "Fraction of pairs in bottom 3",
)

for (j, init) in enumerate(inits)
    x = top3_frac_bt[j]
    y = bottom3_frac_bt[j]

    if isnan(x) || isnan(y)
        continue
    end

    c = color_map[init]

    scatter!(ax, [x], [y];
        color = c,
        markersize = 14,
    )

    text!(ax, init_labels[init];
        position = (x, y),
        align    = (:left, :center),
        fontsize = 10,
        offset   = (6, 0),
    )
end

# median threshold lines
xmed = median(top3_frac_bt[.!isnan.(top3_frac_bt)])
ymed = median(bottom3_frac_bt[.!isnan.(bottom3_frac_bt)])

lines!(ax, [xmed, xmed], [0, 1]; color = (:gray, 0.5), linestyle = :dash)
lines!(ax, [0, 1], [ymed, ymed]; color = (:gray, 0.5), linestyle = :dash)

fig


multi_score = top3_frac_bt .- bottom3_frac_bt

# sort inits by multifunctionality score (best first)
order = sortperm(multi_score)

sorted_scores = multi_score[order]
sorted_inits  = inits[order]
sorted_labels = [init_labels[i] for i in sorted_inits]

fig = Figure(resolution = (900, 900))
ax  = Axis(fig[1, 1],
    ylabel = "Reservoir initializer",
    xlabel = "Multifunctionality score",
    yticks = (1:length(sorted_labels), sorted_labels),
    ylabelsize = 32,
    xlabelsize = 32,
    xticklabelsize = 28,
    yticklabelsize = 28,
)

# horizontal "lollipop" plot
ypos = collect(1:length(sorted_labels))

for (k, y) in enumerate(ypos)
    lines!(ax, [0, sorted_scores[k]], [y, y];
        color = (:gray, 0.7),
        linewidth = 5,
    )
end

vlines!(ax, [0.0]; color = (:gray, 0.5), linestyle = :dash, linewidth = 5)
scatter!(ax, sorted_scores, ypos;
    color      = color_palette[2],
    markersize = 38,
)

fig


save("figures/fig06.eps", fig, dpi=600)
save("figures/fig06.png", fig, dpi=600)

# Additional diagnostics printed to stdout: per-init top3/bottom3 breakdown.
# Uses heat_bt, inits, init_labels, ranks_bt, top3_frac_bt, bottom3_frac_bt,
# multi_score from the multifunctionality-score figure above.

using Printf
using Statistics

n_pairs, n_inits = size(heat_bt)

# convert fractions to counts, e.g. 12/28
top3_count = zeros(Int, n_inits)
bot3_count = zeros(Int, n_inits)
n_valid    = zeros(Int, n_inits)

for j in 1:n_inits
    rj = ranks_bt[:, j]
    valid = rj .> 0
    n_valid[j] = count(valid)
    if n_valid[j] == 0
        continue
    end
    top3_count[j] = count(rj[valid] .<= 3)

    # bottom-3 per pair uses the row-dependent threshold from the score figure
    # above; recovered from bottom3_frac_bt by rounding, exact since that
    # fraction came from integer counts / n_valid.
    bot3_count[j] = round(Int, bottom3_frac_bt[j] * n_valid[j])
end

rows = [(init = inits[j],
         label = init_labels[inits[j]],
         n = n_valid[j],
         top3 = top3_frac_bt[j],
         bot3 = bottom3_frac_bt[j],
         top3_n = top3_count[j],
         bot3_n = bot3_count[j],
         score = multi_score[j]) for j in 1:n_inits if n_valid[j] > 0]

rows_sorted = sort(rows; by = r -> r.score, rev=true)

println("BT multifunctionality score summary (best to worst):")
println("Init   top3_frac  bot3_frac  score    top3_n/n   bot3_n/n")
for r in rows_sorted
    @printf("%-4s   %7.3f    %7.3f   %7.3f   %2d/%2d     %2d/%2d\n",
            r.label, r.top3, r.bot3, r.score, r.top3_n, r.n, r.bot3_n, r.n)
end

println("\nLaTeX rows (label & top3 & bottom3 & score & top3_n/n & bot3_n/n):")
for r in rows_sorted
    @printf("%s & %.3f & %.3f & %.3f & %d/%d & %d/%d \\\\\n",
            r.label, r.top3, r.bot3, r.score, r.top3_n, r.n, r.bot3_n, r.n)
end

k = 3
println("\nTop $(k) by score:")
for r in rows_sorted[1:min(k, length(rows_sorted))]
    @printf("  %s: score=%.3f (top3=%d/%d, bottom3=%d/%d)\n",
            r.label, r.score, r.top3_n, r.n, r.bot3_n, r.n)
end

println("\nBottom $(k) by score:")
for r in reverse(rows_sorted)[1:min(k, length(rows_sorted))]
    @printf("  %s: score=%.3f (top3=%d/%d, bottom3=%d/%d)\n",
            r.label, r.score, r.top3_n, r.n, r.bot3_n, r.n)
end






# Exploratory versions of the enrichment/performance heatmaps below (not
# saved); superseded by the combined two-panel figure further down.
bt_root = "bt_results"

inits_all = ordered_reservoir_inits
inits = filter(init -> isdir(joinpath(bt_root, init)), inits_all)
n_inits = length(inits)

systems_set = Set{String}()
for init in inits
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))
    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        for s in obj["systems"]
            push!(systems_set, String(s))
        end
    end
end
systems = sort(collect(systems_set))
n_sys   = length(systems)

sys_index = Dict(s => i for (i, s) in enumerate(systems))

K = 5  # how many best pairs per init to look at

# rows = systems, cols = inits
enrich_counts = zeros(Float64, n_sys, n_inits)

for (j, init) in enumerate(inits)
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    recs = NamedTuple[]
    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        push!(recs, (file = f,
                     mean = obj["mean_error_overall"],
                     systems = String.(obj["systems"])))
    end

    isempty(recs) && continue
    sorted = sort(recs; by = r -> r.mean)

    topk = sorted[1:min(K, length(sorted))]
    for r in topk
        for s in r.systems
            si = sys_index[s]
            enrich_counts[si, j] += 1
        end
    end
end

# normalize to [0,1]: fraction of appearances in top-K pairs
enrich_frac = enrich_counts ./ (2K)

fig1 = Figure(resolution = (900, 500))
ax1  = Axis(
    fig1[1, 1],
    xlabel = "Reservoir initializer",
    ylabel = "System",
    xticks = (1:n_inits, [init_labels[i] for i in inits]),
    yticks = (1:n_sys, systems),
    xticklabelrotation = π/4,
    ylabelsize = 32,
    xlabelsize = 32,
    xticklabelsize = 28,
    yticklabelsize = 28,
)

hm1 = heatmap!(ax1, enrich_frac';
    colormap = :viridis,
    colorrange = (0.0, maximum(enrich_frac)),
)

Colorbar(fig1[2, 1], hm1;
    vertical   = false,
    height     = 22,
    label      = "Fraction of appearances in top-$K pairs",
)

fig1


sum_err = zeros(Float64, n_sys, n_inits)
cnt_err = zeros(Int,      n_sys, n_inits)

for (j, init) in enumerate(inits)
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        err  = obj["mean_error_overall"]
        sys  = String.(obj["systems"])

        for s in sys
            si = sys_index[s]
            sum_err[si, j] += err
            cnt_err[si, j] += 1
        end
    end
end

perf = fill(NaN, n_sys, n_inits)
for si in 1:n_sys, j in 1:n_inits
    if cnt_err[si, j] > 0
        perf[si, j] = sum_err[si, j] / cnt_err[si, j]
    end
end

eps_val  = 1e-12  # floor to avoid log(0)
perf_log = map(x -> isfinite(x) ? log10(max(x, eps_val)) : NaN, perf)

valid_vals = perf_log[.!isnan.(perf_log)]
crange = (minimum(valid_vals), maximum(valid_vals))

fig2 = Figure(resolution = (900, 500))
ax2  = Axis(
    fig2[1, 1],
    xlabel = "Reservoir initializer",
    ylabel = "System",
    xticks = (1:n_inits, [init_labels[i] for i in inits]),
    yticks = (1:n_sys, systems),
    xticklabelrotation = π/4,
    ylabelsize = 32,
    xlabelsize = 32,
    xticklabelsize = 28,
    yticklabelsize = 28,
)

hm2 = heatmap!(ax2, perf_log';
    colormap   = :viridis,
    colorrange = crange,
)

Colorbar(fig2[2, 1], hm2;
    vertical   = false,
    height     = 22,
    label      = "log₁₀ mean CDE over pairs containing system",
)

fig2





fig = Figure(resolution = (800, 1050))

ax_enrich = Axis(
    fig[1, 1],
    xlabel = "",
    ylabel = "System",
    ylabelvisible=false,
    xticks = (1:length(inits), [init_labels[i] for i in inits]),
    yticks = (1:length(systems), systems),
    xticklabelrotation = π/4,
    xticklabelsvisible = false,
    ylabelsize = 28,
    xlabelsize = 28,
    xticklabelsize = 25,
    yticklabelsize = 25,
)

hm_enrich = heatmap!(ax_enrich, enrich_frac';
    colormap   = :amp,
    colorrange = (0.0, maximum(enrich_frac)),
)

Colorbar(
    fig[1, 2], hm_enrich;
    label      = "Fraction in top-$K pairs",
    vertical   = true,
    width      = 20,
    ticksize   = 15,
    tickwidth  = 3,
    spinewidth = 3,
    tickalign  = 0.5,
    labelsize  = 28,
    ticklabelsize = 26,
)

ax_perf = Axis(
    fig[2, 1],
    xlabel = "Reservoir initializer",
    ylabel = "System",
    ylabelvisible=false,
    xticks = (1:length(inits), [init_labels[i] for i in inits]),
    yticks = (1:length(systems), systems),
    xticklabelrotation = π/4,
    ylabelsize = 28,
    xlabelsize = 28,
    xticklabelsize = 25,
    yticklabelsize = 25,
)

hm_perf = heatmap!(ax_perf, perf_log';
    colormap   = :tempo,
    colorrange = crange,
)

Colorbar(
    fig[2, 2], hm_perf;
    label      = "log₁₀ KLD",
    vertical   = true,
    width      = 20,
    ticksize   = 15,
    tickwidth  = 3,
    spinewidth = 3,
    tickalign  = 0.5,
    labelsize  = 28,
    ticklabelsize = 26,
)

Label(
    fig[1, 1, TopLeft()], "(a)";
    fontsize = 24,
    font     = :bold,
    padding  = (10, 10, 10, 10),
    halign   = :left,
)

Label(
    fig[2, 1, TopLeft()], "(b)";
    fontsize = 24,
    font     = :bold,
    padding  = (10, 10, 10, 10),
    halign   = :left,
)

fig

save("figures/fig07.eps", fig, dpi=600)
save("figures/fig07.png", fig, dpi=600)

# Additional diagnostics printed to stdout: recomputes the enrichment/
# performance matrices above (fig07's inputs) to print per-initializer and
# per-system rankings; not part of the saved figure.

using JSON
using Statistics
using Printf

bt_root = "bt_results"
inits_all = ordered_reservoir_inits
inits = filter(init -> isdir(joinpath(bt_root, init)), inits_all)
n_inits = length(inits)

systems_set = Set{String}()
for init in inits
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))
    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        for s in obj["systems"]
            push!(systems_set, String(s))
        end
    end
end
systems = sort(collect(systems_set))
n_sys   = length(systems)
sys_index = Dict(s => i for (i, s) in enumerate(systems))

# Panel (a): top-K enrichment
K = 5
enrich_counts = zeros(Float64, n_sys, n_inits)

for (j, init) in enumerate(inits)
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    recs = NamedTuple[]
    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        push!(recs, (mean = obj["mean_error_overall"],
                     systems = String.(obj["systems"])))
    end
    isempty(recs) && continue

    sorted = sort(recs; by = r -> r.mean)
    topk = sorted[1:min(K, length(sorted))]

    for r in topk
        for s in r.systems
            enrich_counts[sys_index[s], j] += 1
        end
    end
end

# each top-K set contains 2K system appearances (two systems per pair)
enrich_frac = enrich_counts ./ (2K)

# Panel (b): per-system mean error per initializer
sum_err = zeros(Float64, n_sys, n_inits)
cnt_err = zeros(Int,      n_sys, n_inits)

for (j, init) in enumerate(inits)
    init_dir = joinpath(bt_root, init)
    files = filter(f -> endswith(f, ".json"), readdir(init_dir))

    for f in files
        obj = JSON.parsefile(joinpath(init_dir, f); allownan=true)
        err  = obj["mean_error_overall"]
        sys  = String.(obj["systems"])

        for s in sys
            si = sys_index[s]
            sum_err[si, j] += err
            cnt_err[si, j] += 1
        end
    end
end

perf = fill(NaN, n_sys, n_inits)
for si in 1:n_sys, j in 1:n_inits
    if cnt_err[si, j] > 0
        perf[si, j] = sum_err[si, j] / cnt_err[si, j]
    end
end

eps_val  = 1e-12
perf_log = map(x -> isfinite(x) ? log10(max(x, eps_val)) : NaN, perf)

println("\n=== Panel (a): top-$K enrichment summaries ===")

# (a1) For each initializer: top systems by enrichment fraction
m = 3  # how many to print
println("\nPer initializer: top $(m) systems by fraction in top-$K pairs")
for (j, init) in enumerate(inits)
    col = enrich_frac[:, j]
    ord = sortperm(col; rev=true)
    @printf("  %-6s :", init_labels[init])
    for t in 1:min(m, n_sys)
        si = ord[t]
        @printf("  %s=%.2f", systems[si], col[si])
        if t < min(m, n_sys); print(","); end
    end
    println()
end

# (a2) For each system: which initializers include it most in top-K pairs
println("\nPer system: top $(m) initializers by fraction in top-$K pairs")
for (si, s) in enumerate(systems)
    row = enrich_frac[si, :]
    ord = sortperm(row; rev=true)
    @printf("  %-12s :", s)
    for t in 1:min(m, n_inits)
        j = ord[t]
        @printf("  %s=%.2f", init_labels[inits[j]], row[j])
        if t < min(m, n_inits); print(","); end
    end
    println()
end

println("\n=== Panel (b): system × initializer mean log10(KLD) summaries ===")

# (b1) For each system: best/worst initializer by perf_log (lower is better)
println("\nPer system: best and worst initializer by mean log10(KLD)")
for (si, s) in enumerate(systems)
    row = perf_log[si, :]
    valid = .!isnan.(row)
    idxs = findall(valid)
    vals = row[valid]
    isempty(vals) && continue

    best_local = argmin(vals); best_j = idxs[best_local]
    worst_local = argmax(vals); worst_j = idxs[worst_local]

    @printf("  %-12s  best=%-5s %.3f   worst=%-5s %.3f\n",
            s,
            init_labels[inits[best_j]], row[best_j],
            init_labels[inits[worst_j]], row[worst_j])
end

# (b2) For each initializer: best/worst system by perf_log
println("\nPer initializer: best and worst system by mean log10(KLD)")
for (j, init) in enumerate(inits)
    col = perf_log[:, j]
    valid = .!isnan.(col)
    idxs = findall(valid)
    vals = col[valid]
    isempty(vals) && continue

    best_local = argmin(vals); best_si = idxs[best_local]
    worst_local = argmax(vals); worst_si = idxs[worst_local]

    @printf("  %-6s  best=%-12s %.3f   worst=%-12s %.3f\n",
            init_labels[init],
            systems[best_si], col[best_si],
            systems[worst_si], col[worst_si])
end
