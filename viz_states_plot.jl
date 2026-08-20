# Shared, Makie-only plotting for the ICT reservoir-state figures. Depends
# solely on CairoMakie/Colors/Statistics and the house theme (base/theme.jl
# provides `color_palette`), and consumes a plain data dictionary (see
# figures/data/ict_states_<tag>.json) so figures can be regenerated/restyled
# without any ReservoirComputing or ODE machinery.
#
# One figure, two stacked panels:
#   (a) a representative subset of neurons split into the two system
#       populations (green = pair[1], magenta = pair[2]), each z-scored
#       and overlaid (dashed) with the true coordinate it tracks.
#   (b) a broader set of raw reservoir-neuron trajectories (transient
#       dropped), offset-stacked with no grouping/overlay/normalization,
#       to show the state-space portrait as-is.
#
# Data schema:
#   "pair"       => [sys1, sys2]
#   "init_label" => e.g. "SLFC"
#   "kld","k1","k2","seed"
#   "time"       => [...]                            (panel a x axis)
#   "sys1"       => [ {"label","h","truth"}, ... ]    (best-first)
#   "sys2"       => [ {"label","h","truth"}, ... ]
#   "raw_time"   => [...]                             (panel b x axis)
#   "raw"        => [ {"id","h"}, ... ]                (raw traces)

using CairoMakie, Colors, Statistics

_zscore(v) = (s = std(v); s < 1e-8 ? (v .- mean(v)) : (v .- mean(v)) ./ s)
_f64(x)    = Float64.(x)

function plot_states_figure(data; outname::String,
        amp::Float64=0.42, spacing::Float64=1.0, gap::Float64=1.4)

    pair       = String.(data["pair"])
    tvec       = _f64(data["time"])
    sys1       = data["sys1"]
    sys2       = data["sys2"]
    rawt       = _f64(data["raw_time"])
    raw        = data["raw"]

    # Panel (a) spans data["time"] and (b) spans data["raw_time"]; these
    # differ in length and offset, so crop both to their overlap [t0, t1]
    # so the x scales match.
    t0 = max(first(tvec), first(rawt))
    t1 = min(last(tvec),  last(rawt))
    amask = findall(t -> t0 <= t <= t1, tvec)
    bmask = findall(t -> t0 <= t <= t1, rawt)
    tvec  = tvec[amask]
    rawt  = rawt[bmask]

    green   = RGB(color_palette[3])
    magenta = RGB(color_palette[7])
    slate   = RGB(color_palette[1])
    light   = RGB(0.93, 0.93, 0.93)
    dark    = RGB(0.20, 0.20, 0.20)
    tickgrey = RGB(0.38, 0.38, 0.38)
    blend(c1, c2, t) = RGB((1-t)*c1.r + t*c2.r, (1-t)*c1.g + t*c2.g, (1-t)*c1.b + t*c2.b)

    fig = Figure(resolution = (1300, 1580))

    # Panel (a): grouped and compared to the real trajectories. The axis
    # lives in column 2; column 1 holds the two colored "Tracks <system>"
    # labels that stand in for the y ticks.
    axa = Axis(fig[1, 2];
        title  = "ICT reservoir states:  $(pair[1]) – $(pair[2])",
        titlesize = 46, titlefont = :bold, xticklabelsize = 38,
    )
    # Minimal frame: no box, no y ticks, only bottom (non-mirrored) x ticks
    # and labels in dark grey, no x-axis label.
    hidespines!(axa)
    axa.yticksvisible = false
    axa.yticklabelsvisible = false
    axa.xlabelvisible = false
    axa.xticksmirrored = false
    axa.xtickcolor = tickgrey
    axa.xticklabelcolor = tickgrey
    axa.xtickwidth = 3.5
    axa.xticksize = 18

    # Track the actual plotted extremes so the y-limits give every trajectory
    # headroom; z-scored curves can peak past their block base, which
    # otherwise clips the top/bottom traces.
    ylo_seen = Inf; yhi_seen = -Inf

    # Magenta block (system 2) at the bottom.
    n2 = length(sys2)
    for (j, nrec) in enumerate(reverse(sys2))
        base  = (n2 - j + 1) * spacing
        shade = blend(magenta, light, 0.10 + 0.5 * (j - 1) / max(n2 - 1, 1))
        yt = base .+ amp .* _zscore(_f64(nrec["truth"])[amask])
        yh = base .+ amp .* _zscore(_f64(nrec["h"])[amask])
        lines!(axa, tvec, yt; color = RGBA(magenta, 0.38), linestyle = :dash, linewidth = 3.2)
        lines!(axa, tvec, yh; color = shade, linewidth = 3.8)
        ylo_seen = min(ylo_seen, minimum(yt), minimum(yh))
        yhi_seen = max(yhi_seen, maximum(yt), maximum(yh))
    end

    # Green block (system 1) on top.
    n1  = length(sys1)
    off = n2 * spacing + gap
    for (j, nrec) in enumerate(reverse(sys1))
        base  = off + (n1 - j + 1) * spacing
        shade = blend(green, light, 0.10 + 0.5 * (j - 1) / max(n1 - 1, 1))
        yt = base .+ amp .* _zscore(_f64(nrec["truth"])[amask])
        yh = base .+ amp .* _zscore(_f64(nrec["h"])[amask])
        lines!(axa, tvec, yt; color = RGBA(green, 0.38), linestyle = :dash, linewidth = 3.2)
        lines!(axa, tvec, yh; color = shade, linewidth = 3.8)
        ylo_seen = min(ylo_seen, minimum(yt), minimum(yh))
        yhi_seen = max(yhi_seen, maximum(yt), maximum(yh))
    end

    pad = 0.35 * spacing
    ylo = ylo_seen - pad
    yhi = yhi_seen + pad
    ylims!(axa, ylo, yhi)
    xlims!(axa, first(tvec), last(tvec))

    # Colored y-axis labels (in place of tick labels), one per block, placed
    # in a nested 2-row grid so they sit over their block and can't collide.
    # Row weights: small top spacer for the axis title, then green (top
    # block) and magenta (bottom block), small bottom spacer for x-ticks.
    tl = GridLayout(fig[1, 1])
    Label(tl[2, 1], "Tracks $(pair[1])"; rotation = π/2, color = green,
        font = :bold, fontsize = 33)
    Label(tl[3, 1], "Tracks $(pair[2])"; rotation = π/2, color = magenta,
        font = :bold, fontsize = 33)
    rowsize!(tl, 1, Relative(0.05))
    rowsize!(tl, 2, Relative(0.50))
    rowsize!(tl, 3, Relative(0.42))
    rowgap!(tl, 0)

    # Legend explaining solid vs dashed, replacing the old caption. Wide
    # patch + loose dash so the dashed key clearly reads as a broken line.
    solid_key = LineElement(color = dark, linestyle = :solid, linewidth = 4.2)
    dash_key  = LineElement(color = dark, linestyle = (:dash, :loose), linewidth = 3.6)
    Legend(fig[2, 2], [solid_key, dash_key],
        ["Reservoir neuron", "Best-matched system coordinate"];
        orientation = :horizontal, framevisible = false, labelsize = 32,
        patchsize = (66, 22), tellheight = true, tellwidth = false)

    Label(fig[1, 2, TopLeft()], "(a)"; fontsize = 42, font = :bold,
        padding = (0, 10, 6, 0), halign = :left)

    # Panel (b): more neurons, raw values, transient dropped, overlaid at
    # their true h in [-1, 1] (no offset). One hue, shaded dark -> light.
    axb = Axis(fig[3, 2];
        xlabel = "Autonomous prediction step",
        ylabel = "Reservoir state",
        xlabelsize = 46, ylabelsize = 46,
        xticklabelsize = 38, yticklabelsize = 38,
        spinewidth = 4.5,
        xtickwidth = 3.5, ytickwidth = 3.5,
        xticksize = 18, yticksize = 18,
    )

    traces  = [_f64(nrec["h"])[bmask] for nrec in raw]
    # Global amplitude scale so the (typically small-amplitude) raw states
    # fill the [-1, 1] band, preserving relative amplitudes.
    gmax = maximum(maximum(abs, h) for h in traces; init = 0.0)
    gmax = gmax == 0 ? 1.0 : gmax
    N       = length(traces)
    c_dark  = blend(slate, dark,  0.35)
    c_light = blend(slate, light, 0.78)
    for (i, h) in enumerate(traces)
        t   = N == 1 ? 0.0 : (i - 1) / (N - 1)
        col = blend(c_dark, c_light, t)
        lines!(axb, rawt, h ./ gmax; color = RGBA(col, 0.9), linewidth = 2.6)
    end
    ylims!(axb, -1.05, 1.05)
    xlims!(axb, first(rawt), last(rawt))

    Label(fig[3, 2, TopLeft()], "(b)"; fontsize = 42, font = :bold,
        padding = (0, 10, 6, 0), halign = :left)

    colsize!(fig.layout, 1, Fixed(52))
    rowsize!(fig.layout, 1, Relative(0.52))
    rowsize!(fig.layout, 3, Relative(0.44))
    rowgap!(fig.layout, 6)

    save("figures/$(outname).eps", fig, dpi=600)
    save("figures/$(outname).png", fig, dpi=600)
    return fig
end
