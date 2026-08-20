using CairoMakie, ReservoirComputing, Colors, ColorSchemes
include("theme.jl")

soft_gray = RGB(0.95, 0.95, 0.95)

function scatter_matrix!(ax, A)
    rows, cols = size(A)
    vals = vec(A)
    ccmap = [soft_gray, color_palette[10]]
    vals = map(x -> x == 0.0 ? 1 : 2, vals)

    xs = repeat(1:cols, inner=rows)
    ys = repeat((rows:-1:1), outer=cols)  # flipped so row 1 plots at the top

    scatter!(ax, xs, ys,
        color = vals,
        markersize = 31,
        colormap = ccmap
    )

    ax.aspect = DataAspect()
    xlims!(ax, (-1, 12))
    ylims!(ax, (-1, 12))

    hidespines!(ax)
    hidedecorations!(ax)
end

minimal_inits = [
    delay_line,                  # DL      (1,1)
    selfloop_backward_cycle,     # SLFB    (2,1)
    delayline_backward,          # DLB     (1,2)
    selfloop_delayline_backward, # SLDB    (2,2)
    simple_cycle,                # SC      (1,3)
    selfloop_forwardconnection, # SLFC    (2,3)
    cycle_jumps,                 # CJ      (1,4)
    forward_connection,          # FC      (2,4)
    selfloop_cycle,              # SLC     (1,5)
    double_cycle                 # DC      (2,5)
]

mats = [minit(10, 10) for minit in minimal_inits]

titles = [
    "DL",   "SLFB",
    "DLFB",  "SLDB",
    "SC",   "SLFC",
    "CJ",   "FC",
    "SLC",  "DC"
]

panel_labels = [
    "(a)", "(f)",   # column 1: top, bottom
    "(b)", "(g)",   # column 2
    "(c)", "(h)",   # column 3
    "(d)", "(i)",   # column 4
    "(e)", "(j)"    # column 5
]


fig = Figure(size = (2050, 900))
axes = [Axis(fig[i, j]) for i in 1:2, j in 1:5]

for (k, (i, j)) in enumerate(Iterators.product(1:2, 1:5))
    ax = axes[i, j]
    scatter_matrix!(ax, mats[k])
    ax.title = titles[k]
    ax.width = 350
    ax.height = 350
    ax.titlegap = -22

    Label(fig[i, j, TopLeft()],
          panel_labels[k];
          fontsize = 28,
          font = :bold,
          padding = (2,2,2,2),
          halign = :left,
          alignmode = Inside())
end

fig

save("fig01.png", fig, dpi=600)
save("fig01.eps", fig, dpi=600)

