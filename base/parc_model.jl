using ReservoirComputing: AbstractReservoirRecurrentCell, IntegerType, sample_replicate,
    init_hidden_state, replicate, apply, _apply_seq, _set_readout_weight, _apply_washout, InputType, AbstractEchoStateNetwork,
    _wrap_layers, addreadout!

@concrete struct PAESNCell <: AbstractReservoirRecurrentCell
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_reservoir
    init_input
    init_state
    leak_coefficient
end

function PAESNCell((in_dims, out_dims)::Pair{<:Int,<:Int}, activation=tanh;
    init_bias=ones32, init_reservoir=rand_sparse,
    init_input=scaled_rand, init_state=randn32, leak_coefficient=1.0)
    return PAESNCell(activation, in_dims - 1, out_dims, init_bias, init_reservoir,
        init_input, init_state, leak_coefficient)
end

function ReservoirComputing.initialparameters(rng::AbstractRNG, esn::PAESNCell)
    ps = (input_matrix=esn.init_input(rng, esn.out_dims, esn.in_dims),
        reservoir_matrix=esn.init_reservoir(rng, esn.out_dims, esn.out_dims))
    ps = merge(ps, (bias=esn.init_bias(rng, esn.out_dims),))
    return ps
end

function ReservoirComputing.initialstates(rng::AbstractRNG, esn::PAESNCell)
    return (rng=sample_replicate(rng),)
end

function (esn::PAESNCell)(inp::AbstractArray, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_hidden_state(rng, esn, inp)
    return esn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (esn::PAESNCell)((inp, (hidden_state,))::InputType, ps, st::NamedTuple)
    T = eltype(inp)
    label = first(inp)
    true_inp = inp[2:end]
    candidate_h = esn.activation.(ps.input_matrix * true_inp .+
                                  ps.reservoir_matrix * hidden_state .+ label * ps.bias)
    h_new = (T(1.0) - esn.leak_coefficient) .* hidden_state .+
            esn.leak_coefficient .* candidate_h
    return (h_new, (h_new,)), st
end


@concrete struct PAESN <: AbstractEchoStateNetwork{(:cell, :states_modifiers, :readout)}
    cell
    states_modifiers
    readout
end

function PAESN(in_dims::Int, res_dims::Int,
    out_dims::Int, activation=tanh;
    readout_activation=identity,
    state_modifiers=(),
    kwargs...)
    cell = StatefulLayer(PAESNCell(in_dims => res_dims, activation; kwargs...))
    mods_tuple = state_modifiers isa Tuple || state_modifiers isa AbstractVector ?
                 Tuple(state_modifiers) : (state_modifiers,)
    mods = _wrap_layers(mods_tuple)
    ro = LinearReadout(res_dims => out_dims, readout_activation)
    return PAESN(cell, mods, ro)
end

function ReservoirComputing.initialparameters(rng::AbstractRNG, esn::PAESN)
    ps_cell = initialparameters(rng, esn.cell)
    ps_mods = map(l -> initialparameters(rng, l), esn.states_modifiers) |> Tuple
    ps_ro = initialparameters(rng, esn.readout)
    return (cell=ps_cell, states_modifiers=ps_mods, readout=ps_ro)
end

function ReservoirComputing.initialstates(rng::AbstractRNG, esn::PAESN)
    st_cell = initialstates(rng, esn.cell)
    st_mods = map(l -> initialstates(rng, l), esn.states_modifiers) |> Tuple
    st_ro = initialstates(rng, esn.readout)
    return (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function (esn::PAESN)(inp, ps, st)
    out, st_cell = apply(esn.cell, inp, ps.cell, st.cell)
    out, st_mods = _apply_seq(
        esn.states_modifiers, out, ps.states_modifiers, st.states_modifiers)
    out, st_ro = apply(esn.readout, out, ps.readout, st.readout)
    return out, (cell=st_cell, states_modifiers=st_mods, readout=st_ro)
end

function ReservoirComputing.resetcarry!(rng::AbstractRNG, esn::PAESN, st; init_carry=nothing)
    carry = get(st.cell, :carry, nothing)
    if carry === nothing
        outd = esn.cell.cell.out_dims
        sz = outd
    else
        state = first(carry)
        sz = size(state, 1)
    end

    if init_carry === nothing
        new_state = nothing
    else
        new_state = init_carry(rng, sz, 1)
        new_state = (new_state,)
    end

    new_cell = merge(st.cell, (; carry=new_state))
    return (cell=new_cell, states_modifiers=st.states_modifiers, readout=st.readout)
end

function ReservoirComputing.collectstates(esn::PAESN, data::AbstractMatrix, ps, st::NamedTuple)
    newst = st
    collected = Any[]
    for inp in eachcol(data)
        cell_y, st_cell = apply(esn.cell, inp, ps.cell, newst.cell)
        state_t, st_mods = _apply_seq(
            esn.states_modifiers, cell_y, ps.states_modifiers, newst.states_modifiers)
        push!(collected, copy(state_t))
        newst = (cell=st_cell, states_modifiers=st_mods, readout=newst.readout)
    end
    states = eltype(data).(reduce(hcat, collected))
    @assert !isempty(collected)
    states_raw = reduce(hcat, collected)
    states = eltype(data).(states_raw)
    return states, newst
end

function ReservoirComputing.addreadout!(::PAESN, output_matrix::AbstractMatrix,
    ps::NamedTuple, st::NamedTuple)
    @assert hasproperty(ps, :readout)
    new_readout = _set_readout_weight(ps.readout, output_matrix)
    return (cell=ps.cell,
        states_modifiers=ps.states_modifiers,
        readout=new_readout), st
end

function ReservoirComputing.predict(rc::PAESN,
    steps::Integer, ps, st, label; initialdata::AbstractVector)
    output = zeros(eltype(initialdata), length(initialdata), steps)
    for step in 1:steps
        initialdata = vcat(label, initialdata)
        initialdata, st = apply(rc, initialdata, ps, st)
        output[:, step] = initialdata
    end
    return output, st
end

function train_segmented!(esn::PAESN, xs::Vector{<:AbstractMatrix},
    ys::Vector{<:AbstractMatrix}, ps, st;
    train_method=StandardRidge(0.0),
    washout::Int=0,
    rng=Random.default_rng(),
    return_states::Bool=false)

    @assert length(xs) == length(ys)

    states_all = Vector{Matrix{eltype(xs[1])}}()
    targets_all = Vector{Matrix{eltype(xs[1])}}()
    st_cur = st

    for (x, y) in zip(xs, ys)
        st_cur = resetcarry!(rng, esn, st_cur; init_carry=randn32)
        S, st_cur = collectstates(esn, x, ps, st_cur)
        if washout > 0
            S, y = _apply_washout(S, y, washout)
        end
        push!(states_all, S)
        push!(targets_all, y)
    end
    Scat = reduce(hcat, states_all)
    Ycat = reduce(hcat, targets_all)
    W = train(train_method, Scat, Ycat)
    ps2, st_after = ReservoirComputing.addreadout!(esn, W, ps, st_cur)
    return return_states ? ((ps2, st_after), Scat) : (ps2, st_after)
end
