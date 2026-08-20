using Random: MersenneTwister, AbstractRNG, seed!
using JSON: JSON
using OrdinaryDiffEq, OrdinaryDiffEqFeagin, StatsBase, ChaoticDynamicalSystemLibrary, ReservoirComputing
using ConcreteStructs, ProgressBars, DelimitedFiles
using QuasiMonteCarlo
using StatsBase, Statistics, Tullio
using WeightInitializers: ones32, randn32
using ReservoirComputing: setup

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
include(joinpath(PROJECT_PATH, "base", "constants.jl"))
include(joinpath(PROJECT_PATH, "base", "obt_training.jl"))

seed!(42)

chaotic_data = solve_csystems(SYSTEMS)

_json_matrix(A::AbstractMatrix) = [collect(A[i, :]) for i in axes(A, 1)]

_json_forecasts(forecasts) = [
    [_json_matrix(pred) for pred in forecast_set]
    for forecast_set in forecasts
]

function run_pair_for_minit_obt(
        minit,
        paired_systems,
        chaotic_data;
        rng_seed = 17,
        train_len = TRAIN_LEN,
        predict_len = PREDICT_LEN,
        n_ics = N_ICS,
        n_hp = N_HP,
    )
    rng = MersenneTwister(rng_seed)

    system_names = collect(string.(paired_systems))
    systems_name = join(system_names, "")

    println("[Task $RANK] Running OBT/MFRC $(minit) on $systems_name")

    regs, leaks = latin_hypercube_params(n_hp)

    param_grid = Dict(
        :init_input => [minimal_init(; weight = 0.01, sampling_type = :irrational_sample!)],
        :init_bias => [ones32],
        :init_reservoir => [minit],
        :state_modifiers => [ExtendedSquare()],
        :reg => regs,
        :leak_coefficient => leaks,
    )

    dataset, meta = build_attractors_separated(
        train_len,
        predict_len,
        chaotic_data;
        system_names = system_names,
        eta = 0.2,
        shift = 50,
        as_vector = false,
    )

    input_segments = dataset[1]
    output_segments = dataset[2]
    predict_segments = dataset[3]

    best_params = grid_search_obtesn(
        rng,
        input_segments,
        output_segments,
        param_grid,
        1;
        metric = state_space_divergence,
    )

    println("[Task $RANK] Best params for $systems_name:")
    println(best_params)

    cd_total = zeros(n_ics)
    cd_system1 = zeros(n_ics)
    cd_system2 = zeros(n_ics)

    forecasts = Vector{Any}()

    shift = 50

    for ic in 1:n_ics
        dataset, meta_i = build_attractors_separated(
            train_len,
            predict_len,
            chaotic_data;
            system_names = system_names,
            eta = 0.2,
            shift = shift,
            as_vector = false,
        )

        input_segments = dataset[1]
        output_segments = dataset[2]
        predict_segments = dataset[3]

        output = run_best_obtesn(
            rng,
            best_params,
            input_segments,
            output_segments,
            predict_len,
        )

        push!(forecasts, output)

        cd1 = state_space_divergence(predict_segments[1], output[1])
        cd2 = state_space_divergence(predict_segments[2], output[2])

        cd_system1[ic] = cd1
        cd_system2[ic] = cd2
        cd_total[ic] = (cd1 + cd2) / 2

        println(
            "[Task $RANK] $systems_name IC $ic / $n_ics: ",
            "cd1 = $cd1, cd2 = $cd2, mean = $(cd_total[ic])",
        )

        shift += 50
    end

    mean_overall = mean(cd_total)
    std_overall = std(cd_total)

    obj = Dict(
        "model" => "OBT_MFRC_unlabelled",
        "systems" => system_names,
        "reg" => best_params[:reg],
        "leak_coefficient" => best_params[:leak_coefficient],
        "per_run_error_overall" => collect(cd_total),
        "per_run_error_sys1" => collect(cd_system1),
        "per_run_error_sys2" => collect(cd_system2),
        "mean_error_overall" => mean_overall,
        "std_error_overall" => std_overall,
        "forecasts" => _json_forecasts(forecasts),
        "meta" => Dict(
            "train_len" => train_len,
            "predict_len" => predict_len,
            "n_runs" => n_ics,
            "eta" => 0.2,
            "unlabelled" => true,
        ),
    )

    resdir = joinpath(PROJECT_PATH, "obt_results", string(minit))
    mkpath(resdir)

    fpath = joinpath(resdir, "fixed_$(systems_name).json")

    open(fpath, "w") do io
        JSON.json(io, obj, allownan = true)
    end

    println("[Task $RANK] Saved $fpath")

    return nothing
end

all_jobs = [(minit, pair) for minit in MINIMAL_INITS for pair in PAIR_SYSTEMS]

println("[Task $RANK] Total jobs: ", length(all_jobs))

my_jobs = [all_jobs[i] for i in (RANK + 1):NTASKS:length(all_jobs)]

println("[Task $RANK] Assigned ", length(my_jobs), " jobs")

for (minit, pair) in my_jobs
    job_id_string = string(minit) * "_" * join(string.(pair), "_")
    seed = abs(hash(job_id_string)) % 10_000 + RANK

    run_pair_for_minit_obt(
        minit,
        pair,
        chaotic_data;
        rng_seed = seed,
    )
end

println("[Task $RANK] Done.")
