#!/usr/bin/env julia
# Five-panel Lorenz illustration: (1) Lorenz vs Lorenz (different ICs), then
# four increasingly degraded "predictions". Under each panel we report the
# standard histogram-KLD (no OOB penalty) and the modified KLD with OOB
# penalty. Reuses state_space_divergence and total_variation_attractor from
# base/evaluation.jl.

using Random
using Statistics
using StatsBase
using Distances
using OrdinaryDiffEq
using CairoMakie
using Colors
using ColorSchemes
using ReservoirComputing
using ReservoirComputing: setup
using Printf: @sprintf
include("theme.jl")
include("../base/evaluation.jl")

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

function simulate_lorenz(u0; p=(10.0, 28.0, 8/3), tspan=(0.0, 200.0), dt=0.01)
    prob = ODEProblem(lorenz!, Float64.(u0), tspan, Float64.(collect(p)))
    sol  = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12, saveat=dt)
    X = Array(sol) # 3×T
    return X
end
Random.seed!(42)

# Synthetic "degrading" predictors: each takes the true trajectory Xtrue
# (3×T) and returns a predicted Xpred (3×T) of decreasing forecast quality.
function make_pred_good(data; rng=MersenneTwister(1))
    shift = 300
    train_len = 5000
    predict_len = 3000
    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
    esn = ESN(3, 300, 3; init_reservoir=rand_sparse(; radius=1.2, sparsity=6/300),
        init_input=weighted_init(; scaling = 0.1),
        state_modifiers=NLAT2)
    ps, st = setup(rng, esn)
    ps, st = train!(esn, input_data, target_data, ps, st)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=test[:, 1])
    return output
end

function make_pred_medium(data; rng=MersenneTwister(42))
    shift = 300
    train_len = 5000
    predict_len = 3000
    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
    esn = ESN(3, 300, 3; init_reservoir=rand_sparse(; radius=1.5, sparsity=6/300),
    init_input=weighted_init(; scaling = 0.1),
    state_modifiers=NLAT2)
    ps, st = setup(rng, esn)
    ps, st = train!(esn, input_data, target_data, ps, st)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=test[:, 1])
    return output
end

function make_pred_bad(data; rng=MersenneTwister(4))
    shift = 300
    train_len = 5000
    predict_len = 3000
    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
    # unlike make_pred_good/_medium, deliberately omits init_input/state_modifiers
    esn = ESN(3, 300, 3; init_reservoir=rand_sparse(; radius=1.8, sparsity=6/300))
    ps, st = setup(rng, esn)
    ps, st = train!(esn, input_data, target_data, ps, st)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=test[:, 1])
    return output
end

function make_pred_very_bad(data; rng=MersenneTwister(5))
    shift = 300
    train_len = 5000
    predict_len = 3000
    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test = data[:, (shift + train_len):(shift + train_len + predict_len - 1)]
    # deterministic delay-line reservoir instead of a random one, and no
    # init_input/state_modifiers, for a worst-case forecast
    esn = ESN(3, 300, 3; init_reservoir=delay_line)
    ps, st = setup(rng, esn)
    ps, st = train!(esn, input_data, target_data, ps, st)
    output, st = ReservoirComputing.predict(esn, predict_len, ps, st; initialdata=test[:, 1])
    return output
end

# ---------------------------
# Setup data
# ---------------------------
Random.seed!(42)
rng = MersenneTwister(17)

X1 = simulate_lorenz([1.0, 0.0, 0.0]; tspan=(0.0, 220.0), dt=0.01)
X2 = simulate_lorenz([1.001, 0.0, 0.0]; tspan=(0.0, 220.0), dt=0.01)

burn = 4000  # discard transient, keep a window on the attractor
Tkeep = 12000
Xtrue = X1[:, burn:(burn + Tkeep - 1)]
Xtrue2 = X2[:, burn:(burn + Tkeep - 1)]

Xpred1 = Xtrue2
Xpred2 = make_pred_good(Xtrue)
Xpred3 = make_pred_medium(Xtrue)
Xpred4 = make_pred_bad(Xtrue)
Xpred5 = make_pred_very_bad(Xtrue)

preds = [Xpred1, Xpred2, Xpred3, Xpred4, Xpred5]
titles = [
    "Lorenz vs Lorenz (ΔIC)",
    "Good forecast",
    "Degrading forecast",
    "Bad forecast",
    "Very bad (OOB drift)",
]

function safe_log10(x; eps=1e-12)
    return log10(max(x, eps))
end

n_bins = 30
alpha  = 1e-5
oob_w  = 1e2

kld_plain = Float64[]
kld_mod   = Float64[]
tvar_vals = Float64[]

for Xp in preds
    k0 = state_space_divergence(
        Xtrue, Xp, n_bins;
        alpha = alpha,
        penalize_oob = false,
        oob_weight = oob_w,
    )

    k1 = state_space_divergence(
        Xtrue, Xp, n_bins;
        alpha = alpha,
        penalize_oob = true,
        oob_weight = oob_w,
    )

    tv = total_variation_attractor(
        Xtrue, Xp;
        nbins = 10,
        padding = 0.20,
        normalize = true,
    )

    push!(kld_plain, k0)
    push!(kld_mod,   k1)
    push!(tvar_vals, tv)
end

# Colors: grey for truth; one hue for predictions, with shades (panel 1 darkest).
grey_true = RGB(0.72, 0.72, 0.72)
t_lorenz = color_palette[7]
base = color_palette[5]

blend(c1::RGB, c2::RGB, t::Float64) = RGB((1-t)*c1.r + t*c2.r,
                                         (1-t)*c1.g + t*c2.g,
                                         (1-t)*c1.b + t*c2.b)

# darker → lighter shades for panels 1..5
ts = [0.90, 0.80, 0.65, 0.50]
pred_colors = [t_lorenz] 
more_color = [blend(grey_true, base, t) for t in ts]
pred_colors = vcat(pred_colors, more_color)

# use a shorter segment for clarity in the 3D drawing
Nplot = 2500
Xtrue_plot = Xtrue[:, 1:Nplot]

fig  = Figure(size = (2600, 730))
base = fig[1, 1] = GridLayout()

panels = [base[1, i] = GridLayout() for i in 1:5]
labels = [base[2, i] = GridLayout() for i in 1:5]
for i in 1:5
    colsize!(base, i, Relative(1/5))
end
colgap!(base, 10)

Nplot = 1200

for i in 1:5
    gl = panels[i]

    ax = LScene(gl[1, 1], scenekw = (show_axis = false,))

    Xp_plot = preds[i][:, 1:Nplot]

    lines!(ax, Xtrue_plot[1, :], Xtrue_plot[2, :], Xtrue_plot[3, :];
        color = grey_true, linewidth = 3.5)
    lines!(ax, Xp_plot[1, :], Xp_plot[2, :], Xp_plot[3, :];
        color = pred_colors[i], linewidth = 5)

    cam3d!(ax; lookat = Vec3f(0, 0, 0), eyeposition = Vec3f(-5, 5, 0))

    axis3 = ax.scene.plots[1]
    axis3[:showaxis][]  = false
    axis3[:showticks][] = false
    axis3[:showgrid][]  = false

    k0 = kld_plain[i]
    k1 = kld_mod[i]
    tv = tvar_vals[i]

    txt = "log₁₀ dKLD = $(round(safe_log10(k0), digits=3))\n" *
        "log₁₀ KLD = $(round(safe_log10(k1), digits=3))\n" *
        "TVar = $(round(tv, digits=3))"

    Label(labels[i][1, 1], txt;
        fontsize = 40,
        halign   = :center,
        valign   = :center,
        padding  = (0, 0, 0, 0),
        color = RGB(0.12, 0.12, 0.12),
    )
    Label(gl[1, 1, TopLeft()],
        "($(Char('a' + i - 1)))";
        font = :bold,
        padding = (18, 18, 18, 18),
        fontsize = 42,
        halign = :left,
        valign = :top,
        tellwidth = false,
        tellheight = false,
    )
end

resize_to_layout!(fig)
fig


save("fig03.png", fig; dpi=600)
save("fig03.eps", fig; dpi=600)

println("Saved: fig_lorenz_kld_degradation.png and .eps")
println("Per-panel metrics (plain KLD vs modified KLD*):")
for i in 1:5
    println(rpad(titles[i], 24), "  KLD=", @sprintf("%.3e", kld_plain[i]),
            "  KLD*=", @sprintf("%.3e", kld_mod[i]))
end
