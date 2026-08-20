using CairoMakie, Colors, ColorSchemes, ChaoticDynamicalSystemLibrary, OrdinaryDiffEq, OrdinaryDiffEqFeagin, ColorVectorSpace
include("theme.jl")

csystems = [ChaoticDynamicalSystemLibrary.Aizawa,
    ChaoticDynamicalSystemLibrary.Arneodo,
    ChaoticDynamicalSystemLibrary.Chua,
    ChaoticDynamicalSystemLibrary.GenesioTesi,
    ChaoticDynamicalSystemLibrary.Halvorsen,
    ChaoticDynamicalSystemLibrary.Lorenz,
    ChaoticDynamicalSystemLibrary.Rossler,
    ChaoticDynamicalSystemLibrary.SprottS]

attractor_names = [
    "Aizawa",
    "Arneodo",
    "Chua",
    "GenesioTesi",
    "Halvorsen",
    "Lorenz",
    "Rossler",
    "SprottS",
]

function blend(c1::RGB, c2::RGB, t::Float64)
    return RGB(
        (1-t)*c1.r + t*c2.r,
        (1-t)*c1.g + t*c2.g,
        (1-t)*c1.b + t*c2.b
    )
end

grey = RGB(0.6, 0.6, 0.6)
axis_grey = RGB(0.7, 0.7, 0.7)
grey_title = RGB(0.2, 0.2, 0.2)
ts = reverse(range(0.2, 0.8; length = length(csystems)))

fig = Figure(size = (1500, 1000))

for (i, csystem) in enumerate(csystems)
    prob  = csystem()
    data  = Array(solve(prob, Feagin12(); tspan=(0, 5000), abstol=1e-13, reltol=1e-13))
    data_before = data[:, 1:900]

    row = ceil(Int, i/4)  # 2x4 grid, left to right then down
    col = mod1(i, 4)

    ax = LScene(fig[row, col], scenekw=(show_axis=false,))

    letter = Char('a' + i - 1)
    Label(fig[row, col, TopLeft()],
          "($(letter))";
          font = :bold,
          padding = (2,2,2,2),
          fontsize = 26,
          halign = :left,
          tellwidth = false,
          tellheight = false)

    Label(fig[row, col, Top()],
          attractor_names[i];
          padding = (0, 0, 0, 20),
          fontsize = 31,
          color = grey_title,
          tellwidth = false)

    t = ts[i]  # smaller t = more grey, larger t = more color
    base_color = color_palette[5]
    color_mix = blend(grey, base_color, t)

    lines!(ax, data_before[1, :], data_before[2, :], data_before[3, :];
        color=color_mix, linewidth=3)



    cam3d!(ax; lookat=Vec3f(0,0,0), eyeposition=Vec3f(-4,4,0))

    axis = ax.scene.plots[1]  # internal 3D axis object
    axis[:showaxis][] = false
    axis[:showticks][] = false
    axis[:showgrid][] = false
end

fig

save("figa01.png", fig, dpi=600)
save("figa01.eps", fig, dpi=600)