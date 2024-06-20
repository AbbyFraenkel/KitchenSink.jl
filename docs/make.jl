using Documenter
using KitchenSink

makedocs(
    modules=[KitchenSink],
    sitename="KitchenSink.jl Documentation",
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Guide" => "guide.md"
    ]
)
