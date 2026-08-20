struct SingleSystem end
struct BlendedSystems end

@concrete struct ParameterSystems
    num_systems::Int
end

function run_best_esn(rng, best_params, input_data, output_data, predict_len)
    esn, ps, st = train_esn(rng, input_data, output_data; best_params...)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=input_data[:, end])

    return output
end

function grid_search_esn(rng, input_data, output_data, param_grid, n_folds::Int=1;
        metric=state_space_divergence, val_size = 2500, mode=BlendedSystems())
    best_params = nothing
    best_performance = Inf

    param_combinations = collect(Iterators.product(values(param_grid)...))
    for params in ProgressBar(param_combinations)
        fold_params = Dict{Symbol, Any}(zip(keys(param_grid), params))
        performance = temporal_cross_validation(rng, input_data, output_data, fold_params,
            n_folds, val_size; mode=mode, metric=metric)
        if performance < best_performance
            best_performance = performance
            best_params = fold_params
        end
    end

    if best_params === nothing
        error("Grid search: all hyperparameter combinations gave non-finite performance. Check your metric / data.")
    end

    return best_params
end

function temporal_cross_validation(rng, input_data, output_data, params, n_folds,
        val_size = 2500; mode = BlendedSystems(), metric = state_space_divergence)
    performances = zeros(n_folds)
    for fold in 1:n_folds
        train_index_end = div(fold * size(input_data, 2), n_folds) - val_size
        val_index_start = train_index_end + 1
        val_index_end = min(div(fold * size(input_data, 2), n_folds), train_index_end + val_size)
        train_input = input_data[:, 1:train_index_end]
        train_output = output_data[:, 1:train_index_end]
        val_output = output_data[:, val_index_start:val_index_end]
        esn, ps, st = train_esn(rng, train_input, train_output; params...)
        performance = evaluate_esn(mode, esn, val_output, ps, st; metric=metric, initialdata=train_input[:,end])
        performances[fold] = performance
    end
    avg_performance = mean(performances)

    return avg_performance
end

function evaluate_esn(::SingleSystem, esn, target_output::AbstractArray, ps, st;
        metric=state_space_divergence, initialdata)
    predict_len = size(target_output, 2)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=initialdata)
    acc = metric(target_output, output)

    return acc
end

function evaluate_esn(::BlendedSystems, esn, target_output::AbstractArray, ps, st;
        metric=state_space_divergence, initialdata)
    predict_len = size(target_output, 2)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=initialdata)
    acc1 = metric(target_output[1:3,:], output[1:3,:])
    acc2 = metric(target_output[4:6,:], output[4:6,:])

    return (acc1 + acc2)/2
end

function train_esn(rng::AbstractRNG, input_data::AbstractArray, output_data::AbstractArray;
        kwargs...)
    allkw = (; kwargs...)
    res_size = get(allkw, :res_size, 300)
    reg = get(allkw, :reg, 0.0f0)
    esnkw = Base.structdiff(allkw, (; res_size=0, reg=0.0))
    esn = ESN(size(input_data, 1), res_size, size(input_data, 1);
        esnkw...)
    ps, st = setup(rng, esn)
    ps, st = train!(esn, input_data, output_data, ps, st, StandardRidge(reg))

    return esn, ps, st
end

function latin_hypercube_params(n_samples::Int)
    lb = [-12.0, 0.1]
    ub = [ -1.0, 1.0]
    X = QuasiMonteCarlo.sample(n_samples, lb, ub, LatinHypercubeSample())
    reg_exponents = X[1, :]
    leaks = X[2, :]
    regs = 10.0 .^ reg_exponents
    return regs, leaks
end

