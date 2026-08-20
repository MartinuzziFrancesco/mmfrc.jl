
laplace_smoothing(flat_hist::AbstractVector{T}, alpha::T = zero(T)) where {T} =
    (flat_hist .+ alpha) ./ (sum(flat_hist) .+ alpha .* length(flat_hist))

laplace_smoothing(flat_hist_batch::AbstractMatrix{T}, alpha::T = zero(T)) where {T} =
    (flat_hist_batch .+ alpha) ./
    (sum(flat_hist_batch, dims = 1) .+ alpha .* size(flat_hist_batch, 1))

function _state_space_divergence(true_coords::AbstractMatrix{T}, pred_coords::AbstractMatrix{T}, n_bins::Int;
        alpha::T = T(1e-5), penalize_oob::Bool = true, oob_weight::T = T(1e2)
) where {T<:Real}

    @assert size(true_coords, 2) == size(pred_coords, 2) "dimension mismatch"

    std_true = std(true_coords, dims = 1)
    lo = minimum(true_coords, dims = 1) .- T(0.1) .* std_true
    hi = maximum(true_coords, dims = 1) .+ T(0.1) .* std_true


    if penalize_oob
        min_pred = minimum(pred_coords, dims = 1)
        max_pred = maximum(pred_coords, dims = 1)

        lower_violation = max.(zero(T), lo .- min_pred)
        upper_violation = max.(zero(T), max_pred .- hi)

        span = hi .- lo
        span .= ifelse.(span .== 0, one(T), span)

        # relative violation per dimension (0 = inside, 1 = one span outside, etc.)
        rel_violation = (lower_violation .+ upper_violation) ./ span
        max_rel_violation = maximum(rel_violation)

        if max_rel_violation > 0
            @warn "Some of the predicted attractor lies outside the base attractor by $max_rel_violation"
            return oob_weight * max_rel_violation
        end
    end

    bin_edges = Tuple(range.(lo, hi, n_bins + 1))
    hist_true =
        reshape(fit(Histogram{UInt16}, Tuple(eachcol(true_coords)), bin_edges).weights .|> T, :)
    hist_gen =
        reshape(fit(Histogram{UInt16}, Tuple(eachcol(pred_coords)), bin_edges).weights .|> T, :)

    if sum(hist_gen) == 0
        @warn "Empty generated histogram in state_space_divergence" size_X=size(true_coords) size_Xtilde=size(pred_coords)
        return oob_weight
    end

    p = laplace_smoothing(hist_true, alpha)
    q = laplace_smoothing(hist_gen, alpha)

    return kldivergence(p, q)
end

"""
    state_space_divergence(true, predicted)

"""
function state_space_divergence(true_coords::AbstractMatrix{T}, pred_coords::AbstractMatrix{T},
        n_bins::Int=30; kwargs...) where {T<:Real}
    true_coords  = permutedims(true_coords)
    pred_coords = permutedims(pred_coords)

    return _state_space_divergence(true_coords, pred_coords, n_bins; kwargs...)
end


"""
    attractor_deviation(Xtrue, Xpred;
        nbins=50,
        mins=nothing,
        maxs=nothing,
        padding=0.10,
        normalization=:total,
        outside_bin=true,
        return_occupancy=false,
    )

Compute attractor deviation between a reference attractor `Xtrue`
and a generated/predicted attractor `Xpred`.

The phase-space grid is defined from the reference attractor only.
This is important: if the prediction leaves the true attractor region,
the grid is not expanded to make the prediction look better.

Inputs
------
`Xtrue`, `Xpred` can be either:
- D × N matrices, with columns as points
- N × D matrices, with rows as points

Keyword arguments
-----------------
- `nbins`: integer or length-D vector/tuple of grid bins.
- `mins`, `maxs`: optional fixed grid bounds. If not given, they are
  computed from `Xtrue` only.
- `padding`: fractional padding added around the true-attractor bounds.
  For example, `padding=0.10` expands each dimension by 10%.
- `normalization`:
    - `:total`: mismatch divided by total number of grid cells.
    - `:union`: mismatch divided by occupied union.
    - `:none`: raw mismatch count.
    - `:fumagalli`: 0.5 × raw mismatch count.
- `outside_bin`: if true, points outside the padded reference domain
  occupy one extra outside bin. This penalizes predictions that leave
  the true phase-space domain without clamping them onto boundary cells.
- `return_occupancy`: if true, also return occupancy vectors.

Returns
-------
By default returns a scalar ADev score.

If `return_occupancy=true`, returns:

    score, occ_true, occ_pred

Notes
-----
For your current figure, a good choice is:

    attractor_deviation(Xtrue, Xpred; nbins=50, normalization=:total)

and then plot `100 * ADev`, labeled as `ADev (%)`.

For a Fumagalli-style raw support mismatch, use:

    attractor_deviation(Xtrue, Xpred; nbins=50, normalization=:fumagalli)
"""
function attractor_deviation(
    Xtrue,
    Xpred;
    nbins=50,
    mins=nothing,
    maxs=nothing,
    padding=0.10,
    normalization=:total,
    outside_bin=true,
    return_occupancy=false,
)
    Xt = _points_as_columns(Xtrue)
    Xp = _points_as_columns(Xpred)

    D = size(Xt, 1)
    size(Xp, 1) == D || throw(ArgumentError("dimension mismatch"))

    nb = isa(nbins, Integer) ? fill(nbins, D) : collect(nbins)
    length(nb) == D || throw(ArgumentError("nbins must have length D"))

    mins = isnothing(mins) ? vec(minimum(Xt; dims=2)) : collect(mins)
    maxs = isnothing(maxs) ? vec(maximum(Xt; dims=2)) : collect(maxs)

    length(mins) == D || throw(ArgumentError("mins must have length D"))
    length(maxs) == D || throw(ArgumentError("maxs must have length D"))

    widths = maxs .- mins
    widths[widths .== 0.0] .= 1.0

    mins = mins .- padding .* widths
    maxs = maxs .+ padding .* widths

    occ_true = _occupancy_on_reference_grid(Xt, mins, maxs, nb; outside_bin=outside_bin)
    occ_pred = _occupancy_on_reference_grid(Xp, mins, maxs, nb; outside_bin=outside_bin)

    mismatch = xor.(occ_true, occ_pred)
    union_occ = occ_true .| occ_pred

    n_mismatch = count(mismatch)
    n_union = count(union_occ)
    n_total = length(occ_true)

    score = if normalization == :total
        n_mismatch / n_total
    elseif normalization == :union
        n_union == 0 ? 0.0 : n_mismatch / n_union
    elseif normalization == :none
        n_mismatch
    elseif normalization == :fumagalli
        0.5 * n_mismatch
    else
        throw(ArgumentError("normalization must be :total, :union, :none, or :fumagalli"))
    end

    if return_occupancy
        return score, occ_true, occ_pred
    else
        return score
    end
end


"""
    _points_as_columns(X)

Return trajectory as a D × N matrix.

If the input appears to be N × D, it is transposed.
"""
function _points_as_columns(X)
    A = Matrix(X)

    # Chaotic systems usually have D small and N large, so if rows greatly
    # outnumber columns, assume the input is N × D and transpose it.
    if size(A, 1) > size(A, 2)
        return permutedims(A)
    else
        return A
    end
end


"""
    _occupancy_on_reference_grid(X, mins, maxs, nb; outside_bin=true)

Compute binary occupancy of trajectory `X` on a fixed reference grid.

Unlike the previous implementation, this function does NOT clamp
outside points to boundary cells.

If `outside_bin=true`, one extra cell is appended to represent points
outside the reference grid.
"""
function _occupancy_on_reference_grid(X, mins, maxs, nb; outside_bin=true)
    D, N = size(X)

    n_grid_cells = prod(nb)
    n_total_cells = outside_bin ? n_grid_cells + 1 : n_grid_cells
    outside_index = n_total_cells

    occ = falses(n_total_cells)

    strides = ones(Int, D)
    for d in 2:D
        strides[d] = strides[d - 1] * nb[d - 1]
    end

    widths = maxs .- mins
    widths[widths .== 0.0] .= 1.0

    for n in 1:N
        lin = 1
        outside = false

        for d in 1:D
            u = (X[d, n] - mins[d]) / widths[d]

            if !isfinite(u) || u < 0.0 || u > 1.0
                outside = true
                break
            end

            # u = 1 should go to the last bin, not nb+1
            b = min(floor(Int, u * nb[d]) + 1, nb[d])
            lin += (b - 1) * strides[d]
        end

        if outside
            if outside_bin
                occ[outside_index] = true
            end
        else
            occ[lin] = true
        end
    end

    return occ
end

"""
    total_variation_attractor(Xtrue, Xpred;
        nbins=50,
        mins=nothing,
        maxs=nothing,
        padding=0.10,
        normalize=true,
    )

Compute the total variation distance between the empirical state-space
distributions of a reference trajectory `Xtrue` and generated trajectory
`Xpred`.

Both trajectories are histogrammed on the same grid. By default, the grid is
defined from the reference trajectory `Xtrue`, with optional padding.

The returned value is

    TVar = 1/2 * sum(abs.(p_true .- p_pred))

where `p_true` and `p_pred` are normalized histograms.

Interpretation:
- 0 means identical empirical histograms on the grid.
- 1 means disjoint empirical histogram support.
- Lower is better.

Inputs may be D×N matrices, with columns as points, or N×D matrices,
with rows as points.
"""
function total_variation_attractor(
    Xtrue,
    Xpred;
    nbins=50,
    mins=nothing,
    maxs=nothing,
    padding=0.10,
    normalize=true,
)
    Xt = _points_as_columns(Xtrue)
    Xp = _points_as_columns(Xpred)

    D = size(Xt, 1)
    size(Xp, 1) == D || throw(ArgumentError("dimension mismatch"))

    nb = isa(nbins, Integer) ? fill(nbins, D) : collect(nbins)
    length(nb) == D || throw(ArgumentError("nbins must have length D"))

    mins = isnothing(mins) ? vec(minimum(Xt; dims=2)) : collect(mins)
    maxs = isnothing(maxs) ? vec(maximum(Xt; dims=2)) : collect(maxs)

    length(mins) == D || throw(ArgumentError("mins must have length D"))
    length(maxs) == D || throw(ArgumentError("maxs must have length D"))

    widths = maxs .- mins
    widths[widths .== 0.0] .= 1.0

    mins = mins .- padding .* widths
    maxs = maxs .+ padding .* widths

    h_true = _histogram_on_reference_grid(Xt, mins, maxs, nb)
    h_pred = _histogram_on_reference_grid(Xp, mins, maxs, nb)

    if normalize
        s_true = sum(h_true)
        s_pred = sum(h_pred)

        if s_true == 0 || s_pred == 0
            return 1.0
        end

        p_true = h_true ./ s_true
        p_pred = h_pred ./ s_pred

        return 0.5 * sum(abs.(p_true .- p_pred))
    else
        return 0.5 * sum(abs.(h_true .- h_pred))
    end
end


"""
    _histogram_on_reference_grid(X, mins, maxs, nb)

Return a flattened histogram of trajectory `X` on a fixed grid.

Points outside the reference grid are ignored. This mirrors the idea that TVar
is comparing empirical distributions on the selected reference phase-space
domain.

If you want outside points to be counted, use the version with `outside_bin`
below instead.
"""
function _histogram_on_reference_grid(X, mins, maxs, nb)
    D, N = size(X)

    hist = zeros(Float64, prod(nb))

    strides = ones(Int, D)
    for d in 2:D
        strides[d] = strides[d - 1] * nb[d - 1]
    end

    widths = maxs .- mins
    widths[widths .== 0.0] .= 1.0

    for n in 1:N
        lin = 1
        outside = false

        for d in 1:D
            u = (X[d, n] - mins[d]) / widths[d]

            if !isfinite(u) || u < 0.0 || u > 1.0
                outside = true
                break
            end

            b = min(floor(Int, u * nb[d]) + 1, nb[d])
            lin += (b - 1) * strides[d]
        end

        if !outside
            hist[lin] += 1.0
        end
    end

    return hist
end