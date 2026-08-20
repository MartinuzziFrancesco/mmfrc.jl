using Random: MersenneTwister, AbstractRNG, seed!
using JSON: JSON
using OrdinaryDiffEq, StatsBase, ChaoticDynamicalSystemLibrary, ReservoirComputing
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

using OrdinaryDiffEq, OrdinaryDiffEqFeagin, StatsBase, ChaoticDynamicalSystemLibrary,
      ReservoirComputing, ConcreteStructs, FractalDimensions, ProgressBars,
      DelimitedFiles, DSP

include(joinpath(PROJECT_PATH, "base", "data.jl"))
include(joinpath(PROJECT_PATH, "base", "evaluation.jl"))
include(joinpath(PROJECT_PATH, "base", "bt_training.jl"))
include(joinpath(PROJECT_PATH, "base", "constants.jl"))

seed!(42)

chaotic_data = solve_csystems(SYSTEMS)

function run_pair_for_minit(minit, blended_systems, chaotic_data;
        rng_seed = 17, train_len = TRAIN_LEN, predict_len = PREDICT_LEN,
        n_ics = N_ICS, n_hp = N_HP)
    rng = MersenneTwister(rng_seed)

    systems_name = string(blended_systems[1]) * string(blended_systems[2])
    println("[Task $RANK] Running $(minit) on $(systems_name)")

    regs, leaks = latin_hypercube_params(n_hp)
    param_grid = Dict(
        :init_input => [minimal_init(; weight = 0.01, sampling_type = :irrational_sample!)],
        :init_bias => [ones32],
        :init_reservoir => [minit],
        :state_modifiers => [ExtendedSquare()],
        :reg => regs,
        :leak_coefficient => leaks,
    )

    data, _ = build_pair_separated(train_len, predict_len, chaotic_data;
        system_names = blended_systems)
    best_params = grid_search_esn(rng, data[1], data[2], param_grid, 1; mode = BlendedSystems())

    cd_total = zeros(n_ics)
    cd_system1 = zeros(n_ics)
    cd_system2 = zeros(n_ics)
    forecasts = Vector{Any}()
    shift = 50

    for ic in 1:n_ics
        data, _ = build_pair_separated(train_len, predict_len, chaotic_data;
            system_names = blended_systems, shift = shift)

        output = run_best_esn(rng, best_params, data[1], data[2], predict_len)
        push!(forecasts, output)

        cd1 = state_space_divergence(data[3][1:3, :], output[1:3, :])
        cd2 = state_space_divergence(data[3][4:6, :], output[4:6, :])

        cd_system1[ic] = cd1
        cd_system2[ic] = cd2
        cd_total[ic] = (cd1 + cd2) / 2

        shift += 50
    end

    mean_overall = mean(cd_total)
    std_overall = std(cd_total)

    obj = Dict(
        "systems" => [string(blended_systems[1]), string(blended_systems[2])],
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
            "rows_sys1" => "1:3",
            "rows_sys2" => "4:6",
        ),
    )

    basedir = joinpath(PROJECT_PATH, "bt_results", string(minit))
    mkpath(basedir)

    fpath = joinpath(basedir, "fixed_$(systems_name).json")
    open(fpath, "w") do io
        JSON.json(io, obj, allownan = true)
    end
end

all_jobs = [(minit, blended) for minit in MINIMAL_INITS for blended in PAIR_SYSTEMS]
njobs = length(all_jobs)

println("[Task $RANK] Total jobs: $njobs")

my_jobs = [all_jobs[k] for k in (RANK + 1):NTASKS:njobs]

println("[Task $RANK] Assigned ", length(my_jobs), " jobs")

for (minit, blended) in my_jobs
    job_id_string = string(minit) * "_" * join(blended, "_")
    seed = hash(job_id_string) % 10_000 + RANK
    run_pair_for_minit(minit, blended, chaotic_data; rng_seed = seed)
end

println("[Task $RANK] Done.")
