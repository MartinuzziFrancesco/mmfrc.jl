function run_best_obtesn(
    rng,
    best_params,
    input_segs::Vector{<:AbstractMatrix},
    output_segs::Vector{<:AbstractMatrix},
    predict_len::Integer,
)
    esn, ps, st = train_obtesn(
        rng,
        input_segs,
        output_segs;
        best_params...,
    )

    preds = Matrix{eltype(first(output_segs))}[]

    for Xseg in input_segs
        init = @view Xseg[:, end]

        st_i = resetcarry!(rng, esn, st; init_carry = randn32)

        ŷ, st_i = ReservoirComputing.predict(
            esn,
            predict_len,
            ps,
            st_i;
            initialdata = init,
        )

        push!(preds, ŷ)
    end

    return preds
end

function grid_search_obtesn(
    rng_seed,
    input_segments::Vector{<:AbstractMatrix},
    output_segments::Vector{<:AbstractMatrix},
    param_grid,
    n_folds::Int=3;
    metric=state_space_divergence,
    val_size::Int=1000,
    verbose::Bool=true
)
    println("\n==> Running temporal cross validation with $n_folds folds <==\n")

    keys_vec = collect(keys(param_grid))
    vals_vec = [collect(param_grid[k]) for k in keys_vec]
    combos   = collect(Iterators.product(vals_vec...))

    best_params = nothing
    best_performance = Inf

    for tup in ProgressBar(combos)
        fold_params = Dict{Symbol,Any}(zip(keys_vec, tup))

        perf = temporal_cross_validation_obtesn(
            rng_seed, input_segments, output_segments, fold_params, n_folds;
            val_size=val_size, metric=metric
        )

        if perf < best_performance
            best_performance = perf
            best_params = fold_params
        end
    end

    return best_params
end


function temporal_cross_validation_obtesn(
    rng,
    input_segments::Vector{<:AbstractMatrix},
    output_segments::Vector{<:AbstractMatrix},
    params::Dict,
    n_folds::Int;
    val_size::Int=1000,
    metric=state_space_divergence
)
    @assert n_folds ≥ 1
    @assert length(input_segments) == length(output_segments) > 0

    T_in  = _check_same_lengths(input_segments)
    T_out = _check_same_lengths(output_segments)
    @assert T_in == T_out
    T = T_in

    fold_size = div(T, n_folds)
    @assert val_size ≤ fold_size "val_size=$val_size exceeds fold size=$fold_size (T=$T, n_folds=$n_folds)"

    perfs = zeros(n_folds)

    for fold in 1:n_folds
        fold_end       = div(fold * T, n_folds)
        train_end      = max(1, fold_end - val_size)
        val_start      = train_end + 1
        val_end        = fold_end

        train_range    = 1:train_end
        val_range_in   = val_start:val_end
        val_range_out  = val_start:val_end  # same length for 1-step targets

        train_input    = _slice_segments(input_segments,  train_range)
        train_output   = _slice_segments(output_segments, train_range)
        val_input      = _slice_segments(input_segments,  val_range_in)
        val_output     = _slice_segments(output_segments, val_range_out)

        esn, ps, st = train_obtesn(rng, train_input, train_output; params...)

        initials = _initials_from_valinput(val_input)
        perfs[fold] = evaluate_obtesn(rng, esn, val_output, ps, st;
                                     metric=metric, initialdata=initials)
    end

    return mean(perfs)
end

function evaluate_obtesn(
    rng::AbstractRNG,
    esn,
    target_output,
    ps,
    st;
    metric = compare_corr_dim,
    initialdata,
)
    acc = 0.0

    for (idx, system_data) in enumerate(target_output)
        predict_len = size(system_data, 2)
        id = initialdata[idx]

        st_i = resetcarry!(rng, esn, st; init_carry = randn32)

        output, st_i = ReservoirComputing.predict(
            esn,
            predict_len,
            ps,
            st_i;
            initialdata = id,
        )

        acc += metric(system_data, output)
    end

    return acc / length(target_output)
end

function train_segmented_mfrc!(
    esn,
    input_data::Vector{<:AbstractMatrix},
    output_data::Vector{<:AbstractMatrix},
    ps,
    st;
    train_method = StandardRidge(0.0f0),
    rng,
    washout::Int = 300,
    init_carry = randn32,
)
    @assert length(input_data) == length(output_data) > 0

    state_blocks = Matrix[]
    target_blocks = Matrix[]

    for (X, Y) in zip(input_data, output_data)
        @assert size(X, 2) == size(Y, 2)
        @assert washout < size(X, 2)

        # carry must be reset between systems, otherwise state leaks across tasks
        st_i = resetcarry!(rng, esn, st; init_carry = init_carry)

        raw_states, _ = ReservoirComputing.collectstates(esn, X, ps, st_i)

        keep = (washout + 1):size(X, 2)

        push!(state_blocks, Matrix(@view raw_states[:, keep]))
        push!(target_blocks, Matrix(@view Y[:, keep]))
    end

    states_all = hcat(state_blocks...)
    targets_all = hcat(target_blocks...)

    output_matrix = ReservoirComputing.train(
        train_method,
        states_all,
        targets_all,
    )

    ps, st = ReservoirComputing.addreadout!(esn, output_matrix, ps, st)
    st = merge(st, (; states = states_all))

    return ps, st
end

function train_obtesn(
    rng::AbstractRNG,
    input_data::Vector{<:AbstractMatrix},
    output_data::Vector{<:AbstractMatrix};
    kwargs...,
)
    allkw = (; kwargs...)

    res_size = get(allkw, :res_size, 300)
    reg      = get(allkw, :reg, 0.0f0)
    washout  = get(allkw, :washout, 300)

    esnkw = Base.structdiff(allkw, (; res_size = 0, reg = 0.0f0, washout = 0))

    esn = ESN(3, res_size, 3; esnkw...)

    ps, st = setup(rng, esn)

    ps, st = train_segmented_mfrc!(
        esn,
        input_data,
        output_data,
        ps,
        st;
        train_method = StandardRidge(reg),
        rng = rng,
        washout = washout,
    )

    return esn, ps, st
end

_slice_segments(segs::Vector{<:AbstractMatrix}, r::UnitRange{Int}) =
    [@view seg[:, r] for seg in segs]

function _check_same_lengths(segs::Vector{<:AbstractMatrix})
    T = size(segs[1], 2)
    @assert all(size(A,2) == T for A in segs) "All segments must share the same length"
    return T
end

_initials_from_valinput(val_input::Vector{<:AbstractMatrix}) =
    [@view X[:, 1] for X in val_input]


function latin_hypercube_params(n_samples::Int)
    lb = [-12.0, 0.1]
    ub = [ -1.0, 1.0]
    X = QuasiMonteCarlo.sample(n_samples, lb, ub, LatinHypercubeSample())
    reg_exponents = X[1, :]
    leaks = X[2, :]
    regs = 10.0 .^ reg_exponents
    return regs, leaks
end


