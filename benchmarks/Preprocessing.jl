using BenchmarkTools
using .KSTypes, .Preprocessing, .SpectralMethods, .CoordinateSystems

function benchmark_solve_equation(dim::Int, num_elements::Int, polynomial_degree::Int)
	domain = ntuple(i -> (0.0, 1.0), dim)
	coord_system = KSCartesianCoordinates{dim, Float64}(ntuple(i -> 0.0, dim))
	mesh = generate_initial_mesh(domain, coord_system, ntuple(i -> num_elements, dim), polynomial_degree)

	problem = KSPDEProblem{Float64, dim}((D, u, x, idx) -> sum(-D[i][idx[i], :] ⋅ u for i in 1:dim),  # -Δu
										 x -> sum(sin(π * x[i]) for i in 1:dim),  # Source term: f(x) = Σ sin(πxᵢ)
										 x -> 0.0,  # Dirichlet BC: u = 0 on boundary
										 domain,
										 coord_system)

	@benchmark solve_equation($problem, $mesh)
end

# Run benchmarks
println("1D problem:")
display(benchmark_solve_equation(1, 10, 3))

println("\n2D problem:")
display(benchmark_solve_equation(2, 10, 3))

println("\n3D problem:")
display(benchmark_solve_equation(3, 5, 2))

function benchmark_solvers(A::AbstractArray, b::AbstractArray{T}, solvers::Vector{S}, benchmark_func::Function = @benchmark) where {T <: Real, S <: AbstractKSLinearSolver}
	results = DataFrame(Solver = String[], Time = Float64[], Residual = Float64[])

	for solver in solvers
		benchmark = benchmark_func(solve_linear_system($A, $b, $solver))
		x = solve_linear_system(A, b, solver)
		residual = norm(b - A * x)

		push!(results, (string(solver), median(benchmark.times) / 1e9, residual))
	end

	return results
end
