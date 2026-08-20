
function run_best_paesn(rng, best_params, input_segs::Vector{<:AbstractMatrix},
                      output_segs::Vector{<:AbstractMatrix}, predict_len::Integer)

    esn, ps, st = train_paesn(rng, input_segs, output_segs; best_params...)

    preds = Matrix{eltype(first(output_segs))}[]
    for Xseg in input_segs
        seed = @view Xseg[:, end]
        label = seed[1]
        init  = @view seed[2:end]
        st = resetcarry!(rng, esn, st; init_carry=randn32)
        ŷ, st = ReservoirComputing.predict(esn, predict_len, ps, st, label; initialdata=init)
        push!(preds, ŷ)
    end

    return preds
end

function grid_search_paesn(
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

        perf = temporal_cross_validation_paesn(
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


function temporal_cross_validation_paesn(
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

        esn, ps, st = train_paesn(rng, train_input, train_output; params...)

        initials = _initials_from_valinput(val_input)
        perfs[fold] = evaluate_paesn(rng, esn, val_output, ps, st;
                                     metric=metric, initialdata=initials)
    end

    return mean(perfs)
end

function evaluate_paesn(rng::AbstractRNG, esn::PAESN, target_output, ps, st;
        metric=compare_corr_dim, initialdata)

    acc = 0.0
    for (idx, system_data) in enumerate(target_output)
        label = system_data[1]
        predict_len = size(system_data, 2)
        id = initialdata[idx]
        st = resetcarry!(rng, esn, st; init_carry=randn32)
        output, st = ReservoirComputing.predict(esn, predict_len, ps, st, label; initialdata=id[2:end])
        acc += metric(system_data, output)
    end
    acc = acc/length(target_output)

    return acc
end

function train_paesn(rng::AbstractRNG, input_data::AbstractArray, output_data::AbstractArray;
        kwargs...)
    allkw = (; kwargs...)
    res_size = get(allkw, :res_size, 300)
    reg = get(allkw, :reg, 0.0f0)
    esnkw = Base.structdiff(allkw, (; res_size=0, reg=0.0))
    esn = PAESN(4, res_size, 3;
        esnkw...)
    ps, st = setup(rng, esn)
    ps, st = train_segmented!(esn, input_data, output_data, ps, st; train_method = StandardRidge(reg),
        rng=rng, washout=300)
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
