using Random: Random, MersenneTwister, AbstractRNG, seed!
using JSON: JSON
using OrdinaryDiffEq, OrdinaryDiffEqFeagin, StatsBase, ChaoticDynamicalSystemLibrary, ReservoirComputing
using ConcreteStructs, ProgressBars, DelimitedFiles
using QuasiMonteCarlo
using StatsBase, Statistics, Tullio
using WeightInitializers: ones32, randn32
using ReservoirComputing: setup, apply

const RANK = parse(Int, get(ENV, "SLURM_PROCID", "0"))
const NTASKS = parse(Int, get(ENV, "SLURM_NTASKS", "1"))

println("=== Slurm task $RANK / $NTASKS ===")
println("[ Info ] Threads per worker: ", Threads.nthreads())

const PROJECT_PATH = @__DIR__
import Pkg
Pkg.activate(PROJECT_PATH)
Pkg.instantiate()

include(joinpath(PROJECT_PATH, "base", "data.jl"))
include(joinpath(PROJECT_PATH, "base", "evaluation.jl"))
include(joinpath(PROJECT_PATH, "base", "parc_model.jl"))
include(joinpath(PROJECT_PATH, "base", "parc_training.jl"))
include(joinpath(PROJECT_PATH, "base", "constants.jl"))

seed!(42)

chaotic_data = solve_csystems(SYSTEMS)

function run_pair_for_minit_parc(minit, paired_systems, chaotic_data;
        rng_seed = 17, train_len = TRAIN_LEN, predict_len = PREDICT_LEN,
        n_ics = N_ICS, n_hp = N_HP)
    rng = MersenneTwister(rng_seed)

    systems_name = string(paired_systems[1]) * string(paired_systems[2])
    println("[Task $RANK] Running $(minit) on $systems_name")

    regs, leaks = latin_hypercube_params(n_hp)
    param_grid = Dict(
        :init_input => [minimal_init(; weight = 0.01, sampling_type = :irrational_sample!)],
        :init_bias => [ones32],
        :init_reservoir => [minit],
        :state_modifiers => [ExtendedSquare()],
        :reg => regs,
        :leak_coefficient => leaks,
    )

    data, meta = build_labelled_attractors_separated(
        train_len, predict_len, chaotic_data;
        system_names = paired_systems, beta = 0.4, eta = 0.2,
    )

    best_params = grid_search_paesn(rng, data[1], data[2], param_grid, 1)

    cd_total = zeros(n_ics)
    cd_system1 = zeros(n_ics)
    cd_system2 = zeros(n_ics)
    forecasts = Vector{Any}()
    shift = 50

    for ic in 1:n_ics
        data, _ = build_labelled_attractors_separated(
            train_len, predict_len, chaotic_data;
            system_names = paired_systems, shift = shift,
        )

        output = run_best_paesn(rng, best_params, data[1], data[2], predict_len)
        push!(forecasts, output)

        cd1 = state_space_divergence(data[3][1], output[1])
        cd2 = state_space_divergence(data[3][2], output[2])

        cd_system1[ic] = cd1
        cd_system2[ic] = cd2
        cd_total[ic] = (cd1 + cd2) / 2

        shift += 50
    end

    mean_overall = mean(cd_total)
    std_overall = std(cd_total)

    obj = Dict(
        "systems" => [string(paired_systems[1]), string(paired_systems[2])],
        "reg" => best_params[:reg],
        "leak_coefficient" => best_params[:leak_coefficient],
        "per_run_error_overall" => cd_total,
        "per_run_error_sys1" => cd_system1,
        "per_run_error_sys2" => cd_system2,
        "mean_error_overall" => mean_overall,
        "std_error_overall" => std_overall,
        "forecasts" => forecasts,
        "meta" => Dict(
            "train_len" => train_len,
            "predict_len" => predict_len,
            "n_runs" => n_ics,
        ),
    )

    resdir = joinpath(PROJECT_PATH, "parc_results", string(minit))
    mkpath(resdir)
    fpath = joinpath(resdir, "fixed_$(systems_name).json")
    open(fpath, "w") do io
        JSON.json(io, obj, allownan = true)
    end

    return nothing
end

all_jobs = [(minit, pair) for minit in MINIMAL_INITS for pair in PAIR_SYSTEMS]
println("[Task $RANK] Total jobs: ", length(all_jobs))

my_jobs = [all_jobs[i] for i in (RANK + 1):NTASKS:length(all_jobs)]
println("[Task $RANK] Assigned ", length(my_jobs), " jobs")

for (minit, pair) in my_jobs
    job_id_string = string(minit) * "_" * join(pair, "_")
    seed = hash(job_id_string) % 10_000 + RANK
    run_pair_for_minit_parc(minit, pair, chaotic_data; rng_seed = seed)
end

println("[Task $RANK] Done.")
