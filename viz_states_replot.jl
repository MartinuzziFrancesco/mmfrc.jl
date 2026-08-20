# Regenerates the ICT reservoir-state figures from the saved data in
# figures/data/ict_states_<tag>.json, using Makie only (no ESN/ODE/RC). Edit
# viz_states_plot.jl (styling) or the JSON data (traces) and rerun this to
# restyle without retraining anything.

using CairoMakie
using JSON: JSON

include("base/theme.jl")
include("viz_states_plot.jl")

mkpath("figures")

datadir = "figures/data"
files = sort(filter(f -> startswith(f, "ict_states_") && endswith(f, ".json"),
                    readdir(datadir)))

isempty(files) && error("No data files in $datadir. Run viz_states.jl first to generate them.")

for f in files
    tag  = replace(replace(f, "ict_states_" => ""), ".json" => "")
    data = JSON.parsefile(joinpath(datadir, f))
    plot_states_figure(data; outname = "fig_ict_states_$(tag)")
    println("[replot] $tag")
end

println("[Done] figures regenerated from $datadir")
