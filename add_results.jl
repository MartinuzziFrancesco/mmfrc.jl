using Random: seed!
using JSON: JSON
using Statistics: mean, std
using OrdinaryDiffEq
using OrdinaryDiffEqFeagin
using ChaoticDynamicalSystemLibrary
using ReservoirComputing
using StatsBase

const PROJECT_PATH = @__DIR__

import Pkg
Pkg.activate(PROJECT_PATH)
Pkg.instantiate()

include(joinpath(PROJECT_PATH, "base", "data.jl"))
include(joinpath(PROJECT_PATH, "base", "evaluation.jl"))
include(joinpath(PROJECT_PATH, "base", "constants.jl"))

seed!(42)

# Point this at any metric function defined in base/evaluation.jl.
const NEW_METRIC_FN = total_variation_attractor
const METRIC_KEY = "total_variation"
const METRIC_KWARGS = (; nbins = 10, padding = 0.20)

"""
Ensure forecast is returned as a matrix with expected row count.
Saved forecasts may be nested vectors or transposed.
"""
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

"""
Call the metric with kwargs if provided.
"""
function compute_metric(x, y)
    if isempty(keys(METRIC_KWARGS))
        return NEW_METRIC_FN(x, y)
    else
        return NEW_METRIC_FN(x, y; METRIC_KWARGS...)
    end
end

"""
Update one BT result JSON in place.
Assumes:
- systems = ["Sys1", "Sys2"]
- forecasts[run] is a 6×T prediction
- truth is rebuilt via build_pair_separated(...)
"""
function update_bt_file!(fpath::String, chaotic_data)
    obj = JSON.parsefile(fpath, allownan = true)

    systems = String.(obj["systems"])
    meta = obj["meta"]

    train_len = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])
    n_runs = Int(meta["n_runs"])
    forecasts = obj["forecasts"]

    length(forecasts) == n_runs || @warn "BT: n_runs != length(forecasts) in $fpath"

    metric_total = Float64[]
    metric_sys1 = Float64[]
    metric_sys2 = Float64[]

    shift = 50
    for run_idx in 1:length(forecasts)
        data_pair, _ = build_pair_separated(
            train_len, predict_len, chaotic_data;
            system_names = systems,
            shift = shift,
        )

        truth = data_pair[3]
        forecast = forecast_to_matrix(forecasts[run_idx], 6)

        m1 = compute_metric(truth[1:3, :], forecast[1:3, :])
        m2 = compute_metric(truth[4:6, :], forecast[4:6, :])
        mt = (m1 + m2) / 2

        push!(metric_sys1, Float64(m1))
        push!(metric_sys2, Float64(m2))
        push!(metric_total, Float64(mt))

        shift += 50
    end

    obj["per_run_$(METRIC_KEY)_overall"] = metric_total
    obj["per_run_$(METRIC_KEY)_sys1"] = metric_sys1
    obj["per_run_$(METRIC_KEY)_sys2"] = metric_sys2
    obj["mean_$(METRIC_KEY)_overall"] = mean(metric_total)
    obj["std_$(METRIC_KEY)_overall"] = std(metric_total)

    open(fpath, "w") do io
        JSON.json(io, obj; pretty = 4, allownan = true)
    end

    println("[BT ] Updated: $fpath")
    return nothing
end

"""
Update one PARC result JSON in place.
Assumes:
- systems = ["Sys1", "Sys2"]
- forecasts[run] is indexable as output[1], output[2]
- truth is rebuilt via build_labelled_attractors_separated(...)
"""
function update_parc_file!(fpath::String, chaotic_data)
    obj = JSON.parsefile(fpath, allownan = true)

    systems = String.(obj["systems"])
    meta = obj["meta"]

    train_len = Int(meta["train_len"])
    predict_len = Int(meta["predict_len"])
    n_runs = Int(meta["n_runs"])
    forecasts = obj["forecasts"]

    length(forecasts) == n_runs || @warn "PARC: n_runs != length(forecasts) in $fpath"

    metric_total = Float64[]
    metric_sys1 = Float64[]
    metric_sys2 = Float64[]

    shift = 50
    for run_idx in 1:length(forecasts)
        data_pair, _ = build_labelled_attractors_separated(
            train_len, predict_len, chaotic_data;
            system_names = systems,
            shift = shift,
        )

        truth1 = data_pair[3][1]
        truth2 = data_pair[3][2]

        raw_forecast = forecasts[run_idx]

        pred1 = forecast_to_matrix(raw_forecast[1], size(truth1, 1))
        pred2 = forecast_to_matrix(raw_forecast[2], size(truth2, 1))

        m1 = compute_metric(truth1, pred1)
        m2 = compute_metric(truth2, pred2)
        mt = (m1 + m2) / 2

        push!(metric_sys1, Float64(m1))
        push!(metric_sys2, Float64(m2))
        push!(metric_total, Float64(mt))

        shift += 50
    end

    obj["per_run_$(METRIC_KEY)_overall"] = metric_total
    obj["per_run_$(METRIC_KEY)_sys1"] = metric_sys1
    obj["per_run_$(METRIC_KEY)_sys2"] = metric_sys2
    obj["mean_$(METRIC_KEY)_overall"] = mean(metric_total)
    obj["std_$(METRIC_KEY)_overall"] = std(metric_total)

    open(fpath, "w") do io
        JSON.json(io, obj; pretty = 4, allownan = true)
    end

    println("[PARC] Updated: $fpath")
    return nothing
end

function all_json_files(root::String)
    if !isdir(root)
        @warn "Directory does not exist: $root"
        return String[]
    end

    files = String[]
    for init in readdir(root)
        initdir = joinpath(root, init)
        isdir(initdir) || continue
        for f in readdir(initdir)
            endswith(f, ".json") || continue
            push!(files, joinpath(initdir, f))
        end
    end
    return sort(files)
end

println("[Info] Solving chaotic systems...")
chaotic_data = solve_csystems(SYSTEMS)

bt_root = joinpath(PROJECT_PATH, "bt_results")
parc_root = joinpath(PROJECT_PATH, "parc_results")
obt_root = joinpath(PROJECT_PATH, "obt_results")

bt_files = all_json_files(bt_root)
parc_files = all_json_files(parc_root)
obt_files = all_json_files(obt_root)

println("[Info] Found $(length(bt_files)) BT/ICT files")
println("[Info] Found $(length(parc_files)) PARC files")
println("[Info] Found $(length(obt_files)) original BT files")

for fpath in bt_files
    try
        update_bt_file!(fpath, chaotic_data)
    catch err
        @warn "Failed updating BT file $fpath" exception = (err, catch_backtrace())
    end
end

for fpath in parc_files
    try
        update_parc_file!(fpath, chaotic_data)
    catch err
        @warn "Failed updating PARC file $fpath" exception = (err, catch_backtrace())
    end
end

for fpath in obt_files
    try
        update_parc_file!(fpath, chaotic_data)
    catch err
        @warn "Failed updating original BT file $fpath" exception = (err, catch_backtrace())
    end
end

println("[Done] Added $(METRIC_KEY) to all result JSONs.")
