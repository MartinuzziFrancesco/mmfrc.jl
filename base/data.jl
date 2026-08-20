const CDSL = ChaoticDynamicalSystemLibrary

function normalize_zscore(data)
    dt = fit(ZScoreTransform, data, dims=2)
    new_data = StatsBase.transform(dt, data)
    return new_data
end

function solve_csystems(csystems;
        dt::Float64 = 0.01, abstol::Float64 = 1e-13, reltol::Float64 = 1e-13,
        save_json::Bool = true, json_path::AbstractString = "data/chaotic_systems.json",
        max_T::Int = 20000)

    tmp_data = Dict{String, Matrix{Float32}}()
    lengths  = Int[]

    for system in csystems
        name = string(system)
        println("Now integrating: ", name)

        sol  = solve(system(), Feagin12();
                     tspan=(0.0, 10000.0),
                     abstol=abstol,
                     reltol=reltol)

        data = Float32.(Array(sol))
        data = normalize_zscore(data)

        push!(lengths, size(data, 2))
        tmp_data[name] = data
    end

    common_len = min(minimum(lengths), max_T)

    chaotic_systems = Dict{String, Matrix{Float32}}()
    for (name, data) in tmp_data
        chaotic_systems[name] = data[:, 1:common_len]
    end

    if save_json
        mkpath(dirname(json_path))
        open(json_path, "w") do io
            JSON.print(io, chaotic_systems)
        end
    end

    return chaotic_systems
end

function build_labelled_attractors_separated(
    train_len::Int,
    predict_len::Int,
    chaotic_data;
    Tseg::Int=5000, dt::Float64=0.01, washout::Int=500,
    beta::Real=0.4, eta::Real=0.2,
    system_names::Vector{String}=["Lorenz", "Rossler", "Chen", "SprottS"],
    as_vector::Bool=false,
    shift::Int=50,
)
    n_systems = length(system_names)
    @assert 2 ≤ n_systems ≤ 4 "pass 2, 3, or 4 systems"

    T = eltype(first(values(chaotic_data)))

    patterns = (
        (+1.0, +1.0, +1.0),
        (-1.0, -1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
    )

    shifts = Dict{String,NTuple{3,T}}(
        name => T.(patterns[i]) for (i, name) in enumerate(system_names)
    )

    s0   = washout + shift + 1
    iX1  = s0
    iX2  = s0 + train_len  - 1
    iY1  = iX1 + 1
    iY2  = iX2 + 1
    iT1  = iY2 + 1
    iT2  = iT1 + predict_len - 1

    etaT = T(eta)
    betaT = T(beta)

    if as_vector
        total_len = n_systems * train_len
        data         = Matrix{T}(undef, 4, total_len)
        target_data  = Matrix{T}(undef, 3, total_len)
        predict_data = Matrix{T}(undef, 3, total_len)

        fill_beta = T(0.1)
        seg_lengths = fill(train_len, n_systems)

        pos = 1
        @inbounds @views for (idx, name) in enumerate(system_names)
            full_data = chaotic_data[name]
            @assert size(full_data, 1) == 3
            @assert iT2 ≤ size(full_data, 2) "Not enough time steps in '$name'"

            sx, sy, sz = shifts[name]

            cols = pos:(pos + train_len - 1)

            data[1, cols] .= fill_beta

            @. data[2, cols] = full_data[1, iX1:iX2] + etaT * sx
            @. data[3, cols] = full_data[2, iX1:iX2] + etaT * sy
            @. data[4, cols] = full_data[3, iX1:iX2] + etaT * sz

            @. target_data[1, cols] = full_data[1, iY1:iY2] + etaT * sx
            @. target_data[2, cols] = full_data[2, iY1:iY2] + etaT * sy
            @. target_data[3, cols] = full_data[3, iY1:iY2] + etaT * sz

            @. predict_data[1, cols] = full_data[1, iT1:iT2] + etaT * sx
            @. predict_data[2, cols] = full_data[2, iT1:iT2] + etaT * sy
            @. predict_data[3, cols] = full_data[3, iT1:iT2] + etaT * sz

            pos += train_len
            fill_beta += betaT
        end

    else
        segs         = Vector{Matrix{T}}(undef, n_systems)
        target_vec   = Vector{Matrix{T}}(undef, n_systems)
        predict_vec  = Vector{Matrix{T}}(undef, n_systems)
        seg_lengths  = fill(train_len, n_systems)

        fill_beta = T(0.1)

        @inbounds @views for (idx, name) in enumerate(system_names)
            full_data = chaotic_data[name]
            @assert size(full_data, 1) == 3
            @assert iT2 ≤ size(full_data, 2) "Not enough time steps in '$name'"

            sx, sy, sz = shifts[name]

            seg = Matrix{T}(undef, 4, train_len)
            seg[1, :] .= fill_beta
            @. seg[2, :] = full_data[1, iX1:iX2] + etaT * sx
            @. seg[3, :] = full_data[2, iX1:iX2] + etaT * sy
            @. seg[4, :] = full_data[3, iX1:iX2] + etaT * sz
            segs[idx] = seg

            Y = Matrix{T}(undef, 3, train_len)
            @. Y[1, :] = full_data[1, iY1:iY2] + etaT * sx
            @. Y[2, :] = full_data[2, iY1:iY2] + etaT * sy
            @. Y[3, :] = full_data[3, iY1:iY2] + etaT * sz
            target_vec[idx] = Y

            P = Matrix{T}(undef, 3, predict_len)
            @. P[1, :] = full_data[1, iT1:iT2] + etaT * sx
            @. P[2, :] = full_data[2, iT1:iT2] + etaT * sy
            @. P[3, :] = full_data[3, iT1:iT2] + etaT * sz
            predict_vec[idx] = P

            fill_beta += betaT
        end

        data         = segs
        target_data  = target_vec
        predict_data = predict_vec
    end

    boundaries = cumsum(seg_lengths)
    meta = (
        order      = system_names,
        beta       = betaT,
        eta        = etaT,
        boundaries = boundaries,
        Tseg       = Tseg,
        dt         = dt,
        washout    = washout,
    )

    return (data, target_data, predict_data), meta
end


function build_pair_separated(
    train_len::Int,
    predict_len::Int,
    chaotic_data;
    dt::Float64 = 0.01,
    washout::Int = 300,
    eta::Real = 0.2,
    shift::Int = 0,
    system_names::Vector{String} = ["Lorenz", "Rossler"],
    floatT::Type{T} = Float32,
) where {T<:AbstractFloat}

    @assert length(system_names) == 2 "Pass exactly two systems"

    patterns = (
        (+1.0, +1.0, +1.0),
        (-1.0, -1.0, -1.0),
    )

    shifts = Dict{String,NTuple{3,T}}(
        name => T.(patterns[i]) for (i, name) in enumerate(system_names)
    )

    etaT = T(eta)

    s0   = washout + shift + 1
    iX1  = s0
    iX2  = s0 + train_len  - 1
    iY1  = iX1 + 1
    iY2  = iX2 + 1
    iT1  = iY2 + 1
    iT2  = iT1 + predict_len - 1

    get_data(name::String) = begin
        raw = chaotic_data[name]

        if raw isa AbstractMatrix
            if eltype(raw) === T
                full = raw
            else
                full = T.(raw)
            end
        else
            rows = raw::AbstractVector
            mat  = hcat(rows...)
            full = permutedims(T.(mat))
        end

        @assert size(full, 1) == 3 "Expected 3D system for '$name'"
        full
    end

    data        = Matrix{T}(undef, 6, train_len)
    target_data = Matrix{T}(undef, 6, train_len)
    test_data   = Matrix{T}(undef, 6, predict_len)

    @inbounds for (sys_idx, name) in enumerate(system_names)
        full = get_data(name)
        @assert iT2 ≤ size(full, 2) "Not enough time steps in '$name' for chosen train_len/predict_len"

        (sx, sy, sz) = shifts[name]

        r1 = 3*(sys_idx-1) + 1
        r2 = r1 + 1
        r3 = r1 + 2

        for t in 1:train_len
            jX = iX1 + t - 1
            jY = iY1 + t - 1

            data[r1, t] = full[1, jX] + etaT*sx
            data[r2, t] = full[2, jX] + etaT*sy
            data[r3, t] = full[3, jX] + etaT*sz

            target_data[r1, t] = full[1, jY] + etaT*sx
            target_data[r2, t] = full[2, jY] + etaT*sy
            target_data[r3, t] = full[3, jY] + etaT*sz
        end

        for t in 1:predict_len
            jT = iT1 + t - 1

            test_data[r1, t] = full[1, jT] + etaT*sx
            test_data[r2, t] = full[2, jT] + etaT*sy
            test_data[r3, t] = full[3, jT] + etaT*sz
        end
    end

    meta = (
        systems    = system_names,
        eta        = etaT,
        shifts     = (patterns[1], patterns[2]),
        dt         = dt,
        washout    = washout,
        length_T   = size(data, 2),
        row_blocks = ((1:3), (4:6)),
    )

    return (data, target_data, test_data), meta
end

function combine_systems(xs::AbstractVector, k::Integer)
    n = length(xs)
    k < 0 && throw(ArgumentError("k must be ≥ 0"))
    k > n && return Vector{Vector{eltype(xs)}}()
    out = Vector{Vector{eltype(xs)}}()
    cur = Vector{eltype(xs)}(undef, k)

    function dfs(pos::Int, start::Int)
        if pos > k
            push!(out, copy(cur)); return
        end
        @inbounds for i in start:(n - (k - pos))
            cur[pos] = xs[i]
            dfs(pos + 1, i + 1)
        end
    end

    dfs(1, 1)
    return out
end


function build_attractors_separated(
    train_len::Int,
    predict_len::Int,
    chaotic_data;
    Tseg::Int = 5000,
    dt::Float64 = 0.01,
    washout::Int = 500,
    eta::Real = 0.2,
    system_names::Vector{String} = ["Lorenz", "Rossler", "Chen", "SprottS"],
    as_vector::Bool = false,
    shift::Int = 50,
)
    n_systems = length(system_names)
    @assert 2 ≤ n_systems ≤ 4 "pass 2, 3, or 4 systems"

    T = eltype(first(values(chaotic_data)))

    # Spatial offsets used to separate the attractors in state space, not labels.
    patterns = (
        (+1.0, +1.0, +1.0),
        (-1.0, -1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
    )

    shifts = Dict{String,NTuple{3,T}}(
        name => T.(patterns[i]) for (i, name) in enumerate(system_names)
    )

    s0   = washout + shift + 1

    iX1  = s0
    iX2  = s0 + train_len - 1

    iY1  = iX1 + 1
    iY2  = iX2 + 1

    iT1  = iY2 + 1
    iT2  = iT1 + predict_len - 1

    etaT = T(eta)

    if as_vector
        total_train_len = n_systems * train_len
        total_pred_len  = n_systems * predict_len

        data         = Matrix{T}(undef, 3, total_train_len)
        target_data  = Matrix{T}(undef, 3, total_train_len)
        predict_data = Matrix{T}(undef, 3, total_pred_len)

        train_lengths = fill(train_len, n_systems)
        pred_lengths  = fill(predict_len, n_systems)

        train_pos = 1
        pred_pos  = 1

        @inbounds @views for (idx, name) in enumerate(system_names)
            full_data = chaotic_data[name]
            @assert size(full_data, 1) == 3
            @assert iT2 ≤ size(full_data, 2) "Not enough time steps in '$name'"

            sx, sy, sz = shifts[name]

            train_cols = train_pos:(train_pos + train_len - 1)
            pred_cols  = pred_pos:(pred_pos + predict_len - 1)

            @. data[1, train_cols] = full_data[1, iX1:iX2] + etaT * sx
            @. data[2, train_cols] = full_data[2, iX1:iX2] + etaT * sy
            @. data[3, train_cols] = full_data[3, iX1:iX2] + etaT * sz

            @. target_data[1, train_cols] = full_data[1, iY1:iY2] + etaT * sx
            @. target_data[2, train_cols] = full_data[2, iY1:iY2] + etaT * sy
            @. target_data[3, train_cols] = full_data[3, iY1:iY2] + etaT * sz

            @. predict_data[1, pred_cols] = full_data[1, iT1:iT2] + etaT * sx
            @. predict_data[2, pred_cols] = full_data[2, iT1:iT2] + etaT * sy
            @. predict_data[3, pred_cols] = full_data[3, iT1:iT2] + etaT * sz

            train_pos += train_len
            pred_pos  += predict_len
        end

        boundaries = cumsum(train_lengths)
        pred_boundaries = cumsum(pred_lengths)

    else
        data_vec    = Vector{Matrix{T}}(undef, n_systems)
        target_vec  = Vector{Matrix{T}}(undef, n_systems)
        predict_vec = Vector{Matrix{T}}(undef, n_systems)

        train_lengths = fill(train_len, n_systems)
        pred_lengths  = fill(predict_len, n_systems)

        @inbounds @views for (idx, name) in enumerate(system_names)
            full_data = chaotic_data[name]
            @assert size(full_data, 1) == 3
            @assert iT2 ≤ size(full_data, 2) "Not enough time steps in '$name'"

            sx, sy, sz = shifts[name]

            X = Matrix{T}(undef, 3, train_len)
            @. X[1, :] = full_data[1, iX1:iX2] + etaT * sx
            @. X[2, :] = full_data[2, iX1:iX2] + etaT * sy
            @. X[3, :] = full_data[3, iX1:iX2] + etaT * sz
            data_vec[idx] = X

            Y = Matrix{T}(undef, 3, train_len)
            @. Y[1, :] = full_data[1, iY1:iY2] + etaT * sx
            @. Y[2, :] = full_data[2, iY1:iY2] + etaT * sy
            @. Y[3, :] = full_data[3, iY1:iY2] + etaT * sz
            target_vec[idx] = Y

            P = Matrix{T}(undef, 3, predict_len)
            @. P[1, :] = full_data[1, iT1:iT2] + etaT * sx
            @. P[2, :] = full_data[2, iT1:iT2] + etaT * sy
            @. P[3, :] = full_data[3, iT1:iT2] + etaT * sz
            predict_vec[idx] = P
        end

        data         = data_vec
        target_data  = target_vec
        predict_data = predict_vec

        boundaries = cumsum(train_lengths)
        pred_boundaries = cumsum(pred_lengths)
    end

    meta = (
        order           = system_names,
        eta             = etaT,
        shifts          = shifts,
        boundaries      = boundaries,
        pred_boundaries = pred_boundaries,
        Tseg            = Tseg,
        dt              = dt,
        washout         = washout,
        train_len       = train_len,
        predict_len     = predict_len,
    )

    return (data, target_data, predict_data), meta
end
