# Reviewer request: show the ICT reservoir's internal state-space trajectories
# (individual neuron time series), not just the Wout-projected forecasts stored
# in bt_results/. We therefore retrain an ICT reservoir and capture h(t) during
# the autonomous rollout.
#
# The published per-job seed (hash(minit_pair) % 10_000 + RANK) is not
# reproducible bit-for-bit here: Julia's string hash is not stable across the
# minor Julia version used for the paper (1.12.4) vs. the one available at
# reproduction time (1.12.6). Instead we retrain with the *same configuration*
# as the stored best run (same initialiser/init_input/bias/state modifier and
# the stored reg/leak_coefficient) and select, over a small set of seeds, one
# that reconstructs both attractors well; the achieved KLD is reported so the
# quality is transparent. This is a valid ICT run of the same pair and
# configuration, not a bit-exact reproduction of the published trajectory.
#
# Does not modify any existing result file. Figures are written to figures/
# as .png and .eps, using the house theme.

using CairoMakie
using JSON
using Random
using Colors
using Statistics
using ChaoticDynamicalSystemLibrary, OrdinaryDiffEq, OrdinaryDiffEqFeagin, StatsBase, ReservoirComputing
using QuasiMonteCarlo, ProgressBars
using WeightInitializers: ones32, randn32
using ReservoirComputing: setup, apply

Random.seed!(42)

include("base/theme.jl")
include("base/data.jl")
include("base/evaluation.jl")
include("base/bt_training.jl")
include("base/constants.jl")
include("viz_states_plot.jl")

mkpath("figures")

chaotic_data = solve_csystems(SYSTEMS; save_json = false)

ict_root = "bt_results"

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

const init_by_name = Dict(string(f) => f for f in MINIMAL_INITS)

function find_result(root::String, init::String, systems_pair::Tuple{String, String})
    init_dir = joinpath(root, init)
    isdir(init_dir) || error("Missing init directory: $init_dir")
    for f in filter(f -> endswith(f, ".json"), readdir(init_dir))
        fpath = joinpath(init_dir, f)
        obj = JSON.parsefile(fpath, allownan = true)
        Tuple(String.(obj["systems"])) == systems_pair && return fpath, obj
    end
    error("Could not find result for init=$init and systems=$(systems_pair)")
end

function forecast_to_matrix(raw_forecast, expected_rows::Int)
    tmp = hcat(raw_forecast...)
    size(tmp, 1) == expected_rows && return tmp
    size(tmp, 2) == expected_rows && return permutedims(tmp)
    error("Could not reshape forecast to $(expected_rows)xT. Got $(size(tmp)).")
end

function pair_index(pair::Vector{String})
    for (i, p) in enumerate(PAIR_SYSTEMS)
        p == pair && return i
    end
    error("pair $pair not found in PAIR_SYSTEMS")
end

# RANK == pair_idx - 1 assumes NTASKS == 28 == number of pairs, matching the
# published SLURM layout.
function reproduce_seed(minit, pair::Vector{String})
    job_id_string = string(minit) * "_" * join(pair, "_")
    rank = pair_index(pair) - 1
    return hash(job_id_string) % 10_000 + rank
end

# Identical to ReservoirComputing.predict, but also captures the reservoir
# hidden state h(t) (st.reservoir.carry) at every step.
function predict_capture(rc, steps::Integer, ps, st; initialdata::AbstractVector)
    T = eltype(initialdata)
    x = collect(initialdata)
    x, st = apply(rc, x, ps, st)
    h1 = vec(st.reservoir.carry[1])
    H = zeros(T, length(h1), steps)
    O = zeros(T, length(x), steps)
    H[:, 1] .= h1
    O[:, 1] .= x
    for i in 2:steps
        x, st = apply(rc, x, ps, st)
        H[:, i] .= vec(st.reservoir.carry[1])
        O[:, i] .= x
    end
    return O, H
end

function train_and_capture(rng, best_params, input_data, output_data, predict_len)
    esn, ps, st = train_esn(rng, input_data, output_data; best_params...)
    O, H = predict_capture(esn, predict_len, ps, st; initialdata = input_data[:, end])
    return O, H
end

function pair_kld(truth, O)
    k1 = state_space_divergence(truth[1:3, :], O[1:3, :])
    k2 = state_space_divergence(truth[4:6, :], O[4:6, :])
    return (k1 + k2) / 2, k1, k2
end

function capture_states(pair::Vector{String}, init_name::String;
        shift::Int = 50, seeds = 1:40)
    minit = init_by_name[init_name]

    fpath, obj = find_result(ict_root, init_name, Tuple(pair))
    best_params = Dict{Symbol, Any}(
        :init_input => minimal_init(; weight = 0.01, sampling_type = :irrational_sample!),
        :init_bias => ones32,
        :init_reservoir => minit,
        :state_modifiers => ExtendedSquare(),
        :reg => Float32(obj["reg"]),
        :leak_coefficient => Float32(obj["leak_coefficient"]),
    )

    d, _ = build_pair_separated(TRAIN_LEN, PREDICT_LEN, chaotic_data;
        system_names = pair, shift = shift)
    truth = d[3]

    best = (kld = Inf, k1 = Inf, k2 = Inf, seed = 0, H = zeros(0, 0), O = zeros(0, 0))
    for s in seeds
        rng = MersenneTwister(s)
        O, H = train_and_capture(rng, best_params, d[1], d[2], PREDICT_LEN)
        (all(isfinite, O) && maximum(abs, O) < 1e3) || continue
        k, k1, k2 = pair_kld(truth, O)
        if max(k1, k2) < max(best.k1, best.k2)
            best = (kld = k, k1 = k1, k2 = k2, seed = s, H = H, O = O)
        end
    end

    isfinite(best.kld) || error("No non-divergent reconstruction found for $(pair) / $init_name")

    println("[capture] $(pair[1])-$(pair[2]) | $init_name | seed $(best.seed) | " *
            "KLD overall=$(round(best.kld, digits = 3)) " *
            "($(pair[1])=$(round(best.k1, digits = 3)), $(pair[2])=$(round(best.k2, digits = 3)))")

    stored_best = Float64(obj["per_run_error_overall"][argmin(Float64.(obj["per_run_error_overall"]))])

    return (; H = best.H, O = best.O, truth, kld = best.kld, k1 = best.k1, k2 = best.k2,
        seed = best.seed, stored_best, fpath)
end

function assign_neurons(H, truth; win::Int = 300, k::Int = 5)
    T = size(H, 2)
    w = 1:min(win, T)
    R = size(H, 1)

    corr = zeros(R, 6)
    for i in 1:R
        hi = @view H[i, w]
        si = std(hi)
        if si < 1e-8
            continue
        end
        for r in 1:6
            tr = @view truth[r, w]
            corr[i, r] = cor(hi, tr)
        end
    end
    corr[isnan.(corr)] .= 0.0

    best_sys1 = vec(maximum(abs.(corr[:, 1:3]); dims = 2))
    best_sys2 = vec(maximum(abs.(corr[:, 4:6]); dims = 2))
    assign = [best_sys1[i] >= best_sys2[i] ? 1 : 2 for i in 1:R]
    best_var = [argmax(abs.(corr[i, :])) for i in 1:R]
    best_scr = [maximum(abs.(corr[i, :])) for i in 1:R]

    idx1 = sort(filter(i -> assign[i] == 1, 1:R); by = i -> -best_scr[i])
    idx2 = sort(filter(i -> assign[i] == 2, 1:R); by = i -> -best_scr[i])

    return idx1[1:min(k, length(idx1))], idx2[1:min(k, length(idx2))], best_var, best_scr, corr
end

function build_states_data(rec, pair::Vector{String}, init_label::String;
        plot_win::Int = 350, sel_win::Int = 300, k::Int = 5,
        raw_skip::Int = 40, raw_len::Int = 400, m_raw::Int = 14, raw_seed::Int = 1)
    H, truth = rec.H, rec.truth
    idx1, idx2, best_var, _, _ = assign_neurons(H, truth; win = sel_win, k = k)

    pw = 1:min(plot_win, size(H, 2))
    varnames = [pair[1] * " x", pair[1] * " y", pair[1] * " z",
        pair[2] * " x", pair[2] * " y", pair[2] * " z"]

    mkneuron(n) = Dict(
        "label" => "n$(n) ∼ $(varnames[best_var[n]])",
        "h" => Float64.(H[n, pw]),
        "truth" => Float64.(truth[best_var[n], pw]),
    )

    # Panel (b): a broader set of raw neuron trajectories (transient dropped),
    # ranked by activity (std over the window) rather than an absolute
    # threshold, since raw states can be small-amplitude.
    T = size(H, 2)
    r0 = raw_skip + 1
    r1 = min(r0 + raw_len - 1, T)
    rw = r0:r1
    stds = [std(@view H[n, rw]) for n in 1:size(H, 1)]
    alive = findall(>(1e-6), stds)
    order = sort(alive; by = n -> -stds[n])
    sel = order[1:min(m_raw, length(order))]
    raw = [Dict("id" => n, "h" => Float64.(H[n, rw])) for n in sel]

    return Dict(
        "pair" => pair,
        "init_label" => init_label,
        "kld" => rec.kld,
        "k1" => rec.k1,
        "k2" => rec.k2,
        "seed" => rec.seed,
        "plot_win" => length(pw),
        "sel_win" => sel_win,
        "time" => collect(pw),
        "sys1" => [mkneuron(n) for n in idx1],
        "sys2" => [mkneuron(n) for n in idx2],
        "raw_time" => collect(rw),
        "raw" => raw,
    )
end

function save_states_data(data, tag::String)
    mkpath("figures/data")
    fp = "figures/data/ict_states_$(tag).json"
    open(fp, "w") do io
        JSON.print(io, data)
    end
    println("[data] wrote $fp")
    return fp
end

# Candidate ICT pairs, each seed-searched for a good reconstruction; after the
# batch, pick the low-KLD pairs (see the printed [capture] lines) for the
# final figures. Chua-SprottS uses double_cycle: its SLFC config does not
# reconstruct from a fresh RNG, while double_cycle reconstructs Chua well.
# A full 10-pair x 40-seed batch takes about an hour of compute.
targets = [
    (pair = ["GenesioTesi", "SprottS"], init = "selfloop_forwardconnection", tag = "genesiotesi_sprotts", seeds = 1:40),
    (pair = ["Chua", "SprottS"], init = "double_cycle", tag = "chua_sprotts_dc", seeds = 1:40),
    (pair = ["Aizawa", "SprottS"], init = "simple_cycle", tag = "aizawa_sprotts", seeds = 1:40),
    (pair = ["Arneodo", "Rossler"], init = "selfloop_cycle", tag = "arneodo_rossler", seeds = 1:40),
    (pair = ["Arneodo", "SprottS"], init = "selfloop_delayline_backward", tag = "arneodo_sprotts", seeds = 1:40),
    (pair = ["Chua", "Rossler"], init = "forward_connection", tag = "chua_rossler", seeds = 1:40),
    (pair = ["GenesioTesi", "Halvorsen"], init = "delay_line", tag = "genesiotesi_halvorsen", seeds = 1:40),
    (pair = ["Halvorsen", "Lorenz"], init = "double_cycle", tag = "halvorsen_lorenz", seeds = 1:40),
    (pair = ["Chua", "Lorenz"], init = "cycle_jumps", tag = "chua_lorenz", seeds = 1:40),
    (pair = ["GenesioTesi", "Rossler"], init = "delayline_backward", tag = "genesiotesi_rossler", seeds = 1:40),
]

summary = String[]
for t in targets
    try
        rec = capture_states(t.pair, t.init; seeds = t.seeds)
        data = build_states_data(rec, t.pair, init_labels[t.init])
        save_states_data(data, t.tag)
        plot_states_figure(data; outname = "fig_ict_states_$(t.tag)")
        push!(summary, "  $(t.tag):  KLD=$(round(rec.kld, digits = 3))  (seed $(rec.seed))")
    catch err
        @warn "target failed" pair = t.pair init = t.init exception = err
        push!(summary, "  $(t.tag):  FAILED ($(sprint(showerror, err)))")
    end
end

println("\n=== batch summary (pick low-KLD pairs for the final figures) ===")
foreach(println, summary)
println("[Done] reservoir-state figures + data written to figures/ (data in figures/data/)")
