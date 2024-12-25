using Documenter
using KitchenSink

include("pages.jl")
makedocs(
    sitename = "KitchenSink Documentation",
    format = [
        Documenter.HTML(),  # HTML format for web documentation
        Documenter.PDF()    # Optional PDF documentation (requires LaTeX)
    ],
    pages = [
        "Home" => "index.md",         # Home page
        "API" => "api.md",            # API Reference
        "Guides" => "guide.md",       # General guides
        "Theory" => "theory.md",      # Theoretical overview
        "Tutorials" => "tutorials/index.md"  # Tutorials and examples
    ],
    deploydocs = [
        repo = "AbbyFraenkel/KitchenSink",  # Replace with your GitHub username/repo
        target = "gh-pages"                 # Deployment target branch
    ]
)
