push!(LOAD_PATH,"../src/")
using Documenter, QuantumBilliards
makedocs(sitename="QuantumBilliards.jl",
pages = [
    "index.md",
    "tutorial.md",
    "API.md"
],
format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    size_threshold = 1_000_000,
    size_threshold_warn = 500_000,
)

)

deploydocs(
    repo = "github.com/Quantum-Chaos-Julia/QuantumBilliards.jl.git",
)
