# Runs every ```julia block of README.md in a fresh Julia process inside a temporary sandbox
# directory, so the documented examples cannot silently rot.

function readme_julia_blocks(path::AbstractString)
    blocks = String[]
    current = nothing
    for line in eachline(path)
        if current === nothing
            occursin(r"^```julia\s*$", line) && (current = String[])
        elseif startswith(line, "```")
            push!(blocks, join(current, '\n') * "\n")
            current = nothing
        else
            push!(current, line)
        end
    end
    current === nothing || error("Unterminated code fence in $path")
    return blocks
end

@testset "README examples" begin
    readme = normpath(joinpath(@__DIR__, "..", "..", "..", "README.md"))
    blocks = readme_julia_blocks(readme)
    @test !isempty(blocks)

    mktempdir() do sandbox
        for (i, code) in enumerate(blocks)
            script = joinpath(sandbox, "readme_example_$i.jl")
            write(script, code)
            out = joinpath(sandbox, "readme_example_$i.log")
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --color=no $script`
            proc = run(pipeline(Cmd(cmd; dir=sandbox, ignorestatus=true); stdout=out, stderr=out))
            ok = success(proc)
            ok || @error "README example $i failed" output = read(out, String)
            @test ok
        end
    end
end
