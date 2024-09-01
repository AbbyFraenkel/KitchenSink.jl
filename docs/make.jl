using Documenter
using KitchenSink

makedocs(
    modules = [
        AdaptiveMethods,
        CommonMethods,
        CoordinateSystems,
        ErrorEstimation,
        IntergridOperators,
        KSTypes,
        LinearSolvers,
        MultiLevelMethods,
        Optimization,
        Preconditioners,
        Preprocessing,
        ProblemTypes,
        SpectralMethods,
        TimeStepping
    ],
    sitename = "KitchenSink Documentation",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Guides" => "guide.md",
        "Tutorials" => "tutorials/index.md",
    ]
)
