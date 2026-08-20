const CSYSTEMS = ["Aizawa", "Arneodo", "Chua", "GenesioTesi", "Halvorsen",
    "Lorenz", "Rossler", "SprottS"]

const SYSTEMS = [ChaoticDynamicalSystemLibrary.Aizawa,
    ChaoticDynamicalSystemLibrary.Arneodo,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Halvorsen,
    ChaoticDynamicalSystemLibrary.Lorenz,
    ChaoticDynamicalSystemLibrary.Rossler,
    ChaoticDynamicalSystemLibrary.SprottS]

const PAIR_SYSTEMS = combine_systems(CSYSTEMS, 2)

const MINIMAL_INITS = [
    simple_cycle,
    cycle_jumps,
    delay_line,
    delayline_backward,
    double_cycle,
    forward_connection,
    selfloop_cycle,
    selfloop_delayline_backward,
    selfloop_backward_cycle,
    selfloop_forwardconnection,
]

const TRAIN_LEN = 10000
const PREDICT_LEN = 2500
const N_ICS = 10
const N_HP = 25
