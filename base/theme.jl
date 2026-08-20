fullpage_theme = Theme(
    Axis = (
        ticklabelsize=32,
        xticklabelsize = 28,
        yticklabelsize = 28,
        xlabelsize = 34,
        ylabelsize = 34,
        titlesize = 41,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
        xtickalign = 1.0,
        ytickalign = 1.0,
        xticksmirrored = true,
        yticksmirrored = true,
        titlefont = :regular,
        xticksize = 14,
        yticksize = 14,
        xtickwidth = 2,
        ytickwidth = 2,
        spinewidth = 3,
    ),
    fontsize=34,
    backgroundcolor = RGBf(1.0, 1.0, 1.0),
    fonts = (; regular = "Liberation Sans", bold = "Liberation Sans Bold")
)
CairoMakie.activate!(type = "svg")
set_theme!(fullpage_theme)
color_palette = Makie.colorschemes[:seaborn_muted]
