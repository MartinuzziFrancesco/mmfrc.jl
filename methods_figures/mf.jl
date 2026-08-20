using CairoMakie, Colors, ColorSchemes, ChaoticDynamicalSystemLibrary, OrdinaryDiffEq, OrdinaryDiffEqAdamsBashforthMoulton, ColorVectorSpace
include("theme.jl")

function blend(c1::RGB, c2::RGB, t::Float64)
    return RGB(
        (1-t)*c1.r + t*c2.r,
        (1-t)*c1.g + t*c2.g,
        (1-t)*c1.b + t*c2.b
    )
end

csystem = ChaoticDynamicalSystemLibrary.Lorenz
prob = csystem()

data = Array(solve(prob, ABM54(); dt=0.005, tspan=(0, 200)))
data_before = data[:, 1:3000]
data_after = data[:, 3000:6000]


fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

grey = RGB(0.8, 0.8, 0.8)
color_mix = blend(grey, color_palette[1], 0.25)

lines!(ax, data_before[1, :], data_before[2, :], data_before[3, :];
    color=color_mix, linewidth=4)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_lorenz_ps_before.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_before[1, 1:2500])

lines!(axes[1], x, data_before[1, 1:2500], color=color_mix, linewidth=10)
lines!(axes[2], x, data_before[2, 1:2500], color=color_mix, linewidth=10)
lines!(axes[3], x, data_before[3, 1:2500], color=color_mix, linewidth=10)

fig

save("fig02bt_lorenz_coords_before.png", fig, dpi=600)








fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

lines!(ax, data_after[1, :], data_after[2, :], data_after[3, :];
    color=color_palette[1], linewidth=3)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_lorenz_ps_after.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_after[1, 1:2500])

lines!(axes[1], x, data_after[1, 1:2500], color=color_palette[1], linewidth=10)
lines!(axes[2], x, data_after[2, 1:2500], color=color_palette[1], linewidth=10)
lines!(axes[3], x, data_after[3, 1:2500], color=color_palette[1], linewidth=10)

fig

save("fig02bt_lorenz_coords_after.png", fig, dpi=600)











csystem = ChaoticDynamicalSystemLibrary.Halvorsen
prob = csystem()

data = Array(solve(prob, ABM54(); dt=0.005, tspan=(0, 200)))
data_before = data[:, 1:5000]
data_after = data[:, 5000:10000]


fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

grey = RGB(0.8, 0.8, 0.8)
color_mix = blend(grey, color_palette[2], 0.25)

lines!(ax, data_before[1, :], data_before[2, :], data_before[3, :];
    color=color_mix, linewidth=3)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_halvorsen_ps_before.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_before[1, 1:2500])

lines!(axes[1], x, data_before[1, 1:2500], color=color_mix, linewidth=10)
lines!(axes[2], x, data_before[2, 1:2500], color=color_mix, linewidth=10)
lines!(axes[3], x, data_before[3, 1:2500], color=color_mix, linewidth=10)

fig

save("fig02bt_halvorsen_coords_before.png", fig, dpi=600)








fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

lines!(ax, data_after[1, :], data_after[2, :], data_after[3, :];
    color=color_palette[2], linewidth=3)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_halvorsen_ps_after.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_after[1, 1:2500])

lines!(axes[1], x, data_after[1, 1:2500], color=color_palette[2], linewidth=10)
lines!(axes[2], x, data_after[2, 1:2500], color=color_palette[2], linewidth=10)
lines!(axes[3], x, data_after[3, 1:2500], color=color_palette[2], linewidth=10)

fig

save("fig02bt_halvorsen_coords_after.png", fig, dpi=600)







csystem = ChaoticDynamicalSystemLibrary.Aizawa
prob = csystem()

data = Array(solve(prob, ABM54(); dt=0.005, tspan=(0, 200)))
data_before = data[:, 1:10000]
data_after = data[:, 10000:20000]

grey = RGB(0.8, 0.8, 0.8)
color_mix = blend(grey, color_palette[3], 0.25)

fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

lines!(ax, data_before[1, :], data_before[2, :], data_before[3, :];
    color=color_mix, linewidth=3)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_aizawa_ps_before.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_before[1, 1:4500])

lines!(axes[1], x, data_before[1, 1:4500], color=color_mix, linewidth=10)
lines!(axes[2], x, data_before[2, 1:4500], color=color_mix, linewidth=10)
lines!(axes[3], x, data_before[3, 1:4500], color=color_mix, linewidth=10)

fig

save("fig02bt_aizawa_coords_before.png", fig, dpi=600)





fig = Figure(size = (500, 500))
ax = LScene(fig[1, 1], scenekw=(show_axis=false,))

lines!(ax, data_after[1, :], data_after[2, :], data_after[3, :];
    color=color_palette[3], linewidth=3)
cam3d!(ax;
    lookat=Vec3f(0, 1, 1), eyeposition=Vec3f(1, 1, 1))
axis = ax.scene.plots[1]  # the internal 3D axis object
axis[:showaxis][] = false
axis[:showticks][] = false
axis[:showgrid][] = false

fig

save("fig02bt_aizawa_ps_after.png", fig, dpi=600)

fig = Figure(size = (500, 500))

axes = [
    Axis(fig[i, 1],
        rightspinevisible=false,
        leftspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
        yticksvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false
    ) for i in 1:3
]

x = 1:length(data_after[1, 1:4500])

lines!(axes[1], x, data_after[1, 1:4500], color=color_palette[3], linewidth=10)
lines!(axes[2], x, data_after[2, 1:4500], color=color_palette[3], linewidth=10)
lines!(axes[3], x, data_after[3, 1:4500], color=color_palette[3], linewidth=10)

fig

save("fig02bt_aizawa_coords_after.png", fig, dpi=600)