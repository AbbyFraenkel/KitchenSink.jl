using ..KSTypes
using Test

@testset "KSTypes Tests" begin
    @testset "Concrete Types Tests" begin
        @testset "KSPoint Tests" begin
            @testset "Valid Cases" begin
                point = KSPoint([1.0, 2.0], 0.5)
                @test point.coordinates == [1.0, 2.0]
                @test point.weight == 0.5

                point_no_weight = KSPoint([1.0, 2.0], nothing)
                @test point_no_weight.weight === nothing
            end

            @testset "Invalid Cases" begin
                @test_throws MethodError KSPoint("invalid", 0.5)
                @test_throws MethodError KSPoint([1.0, 2.0], "invalid")
            end
        end

        @testset "KSBasisFunction Tests" begin
            @testset "Valid Cases" begin
                basis_func(x) = x^2
                basis_function = KSBasisFunction(1, basis_func, true, 2)
                @test basis_function.id == 1
                @test basis_function.function_handle(2) == 4
                @test basis_function.is_removable == true
                @test basis_function.degree == 2
            end

            @testset "Invalid Cases" begin
                @test_throws MethodError KSBasisFunction("1", x -> x^2, true, 2)
                @test_throws MethodError KSBasisFunction(1, "not_a_function", true, 2)
            end
        end

        @testset "KSElement Tests" begin
            # Create example points
            point1 = KSPoint([1.0, 2.0], 0.5)
            point2 = KSPoint([2.0, 3.0], 0.5)
            point3 = KSPoint([3.0, 4.0], 0.5)

            # Create example basis functions
            basis_func(x) = x^2
            basis_function1 = KSBasisFunction(1, basis_func, true, 2)
            basis_function2 = KSBasisFunction(1, basis_func, true, 2)

            # Create an example element
            element = KSElement(
                1,
                [point1, point3, point2],
                [basis_function1, basis_function2],
                nothing,
                nothing,
                nothing,
                0,
                1,
                0.0,
                [rand(2, 2)]
            )

            # Test field assignments
            @test element.id == 1
            @test element.points == [point1, point3, point2]
            @test element.basis_functions == [basis_function1, basis_function2]
            @test element.neighbors == nothing
            @test element.parent == nothing
            @test element.children == nothing
            @test element.level == 0
            @test element.polynomial_degree == 1
            @test element.error_estimate == 0.0

            @testset "Invalid Cases" begin
                @test_throws MethodError KSElement("1", [point1, point2], [basis_function1], nothing, nothing, nothing, 0, 2, 0.0, [rand(2, 2)])
            end
        end

        @testset "KSProblem Tests" begin
            @testset "Valid Cases" begin
                cartesian = KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0))
                problem = KSProblem(x -> x^2, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), x -> x == 0, cartesian)
                @test problem.equation(2) == 4
                @test problem.domain == ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
                @test problem.coordinate_system == cartesian
            end

            @testset "Invalid Cases" begin
                cartesian = KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0))
                @test_throws MethodError KSProblem("invalid", ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), x -> x == 0, cartesian)
            end
        end
    end

    @testset "Coordinate System Tests" begin
        @testset "KSCartesianCoordinates Tests" begin
            coords = KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0))
            @test length(coords.coordinates) == 3
            @test coords.coordinates[1] === 1.0
            @test coords.coordinates[2] === 2.0
            @test coords.coordinates[3] === 3.0
        end

        @testset "KSPolarCoordinates Tests" begin
            polar = KSPolarCoordinates(1.0, π / 4)
            @test polar.r === 1.0
            @test polar.θ === π / 4
        end

        @testset "KSSphericalCoordinates Tests" begin
            spherical = KSSphericalCoordinates(1.0, π / 4, π / 6)
            @test spherical.r === 1.0
            @test spherical.θ === π / 4
            @test spherical.φ === π / 6
        end

        @testset "KSCylindricalCoordinates Tests" begin
            cylindrical = KSCylindricalCoordinates(1.0, π / 4, 5.0)
            @test cylindrical.r === 1.0
            @test cylindrical.θ === π / 4
            @test cylindrical.z === 5.0
        end
    end

    @testset "KSSolver Tests" begin
        options = KSSolverOptions{Float64}(100, 1e-6, true, 3, 0.1, 10, 2)
        solver = KSSolver{Float64}(options, Float64[], Float64[])

        @test solver.options.max_iterations == 100
        @test solver.options.tolerance == 1e-6
        @test solver.options.adaptive == true
        @test_throws MethodError KSSolver{Int}(options, Int[], Int[])  # Example invalid case
    end

    @testset "Boundary Conditions Tests" begin
        @testset "KSDirichletBC Tests" begin
            @testset "Valid Cases" begin
                dbc = KSDirichletBC{Float64}(x -> x^2, x -> x < 0.5)
                @test dbc.value(0.25) == 0.0625
                @test dbc.boundary(0.25) == true
            end

            @testset "Invalid Cases" begin
                @test_throws TypeError KSDirichletBC{String}(x -> x^2, x -> x < 0.5)
            end

            @testset "Edge Cases" begin
                dbc = KSDirichletBC{Float64}(x -> 0, x -> true)
                @test dbc.value(1.0) == 0
                @test dbc.boundary(-1.0) == true
            end

            @testset "Complex Cases" begin
                dbc = KSDirichletBC{Float64}(x -> sin(x) + cos(x), x -> x >= 0 && x <= π)
                @test dbc.value(π / 4) ≈ sin(π / 4) + cos(π / 4)
                @test dbc.boundary(π / 2) == true
                @test dbc.boundary(π * 1.5) == false
            end
        end

        @testset "KSNeumannBC Tests" begin
            @testset "Valid Cases" begin
                nbc = KSNeumannBC{Float64}(x -> 2x, x -> x > 0)
                @test nbc.flux(1.0) == 2.0
                @test nbc.boundary(1.0) == true
            end

            @testset "Invalid Cases" begin
                @test_throws TypeError KSNeumannBC{String}(x -> 2x, x -> x > 0)
            end

            @testset "Edge Cases" begin
                nbc = KSNeumannBC{Float64}(x -> 0, x -> true)
                @test nbc.flux(0.0) == 0
                @test nbc.boundary(0.0) == true
                nbc_zero_boundary = KSNeumannBC{Float64}(x -> 2x, x -> false)
                @test nbc_zero_boundary.boundary(0.0) == false
            end

            @testset "Complex Cases" begin
                nbc = KSNeumannBC{Float64}(x -> sin(x) + cos(x), x -> x >= 0 && x <= π)
                @test nbc.flux(π / 4) ≈ sqrt(2)
                @test nbc.boundary(π / 2) == true
                @test nbc.boundary(π * 1.5) == false
            end
        end

        @testset "KSRobinBC Tests" begin
            @testset "Valid Cases" begin
                rbc = KSRobinBC{Float64}(x -> x, x -> 2x, x -> 3x, x -> x < 1.0)
                @test rbc.a(0.5) == 0.5
                @test rbc.b(0.5) == 1.0
                @test rbc.c(0.5) == 1.5
                @test rbc.boundary(0.5) == true
            end

            @testset "Invalid Cases" begin
                @test_throws TypeError KSRobinBC{String}(x -> x, x -> 2x, x -> 3x, x -> x < 1.0)
            end

            @testset "Edge Cases" begin
                rbc = KSRobinBC{Float64}(x -> 0, x -> 0, x -> 0, x -> false)
                @test rbc.a(1.0) == 0
                @test rbc.b(1.0) == 0
                @test rbc.c(1.0) == 0
                @test rbc.boundary(1.0) == false
            end

            @testset "Complex Cases" begin
                rbc = KSRobinBC{Float64}(x -> exp(x), x -> log(x + 1), x -> sqrt(x), x -> x >= 0 && x <= 1)
                @test rbc.a(0.5) ≈ exp(0.5)
                @test rbc.b(0.5) ≈ log(1.5)
                @test rbc.c(0.5) ≈ sqrt(0.5)
                @test rbc.boundary(0.5) == true
                @test rbc.boundary(1.5) == false
            end
        end
    end

    @testset "KSBVDAEProblem Tests" begin
        @testset "Valid Cases" begin
            f = (t, y, yp) -> yp - y
            g = (t, y) -> y^2 - 1
            bc = (ya, yb) -> ya + yb
            tspan = (0.0, 1.0)
            y0 = [0.5]
            algebraic_vars = [false]

            bvdae_problem = KSBVDAEProblem(f, g, bc, tspan, y0, algebraic_vars, y0)

            @test bvdae_problem isa AbstractKSBVDAEProblem{Float64,1}
            @test bvdae_problem.tspan == (0.0, 1.0)
            @test bvdae_problem.y0 == [0.5]
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSBVDAEProblem("invalid", "invalid", "invalid", ("invalid", "invalid"), ["invalid"], ["invalid"], ["invalid"])
        end

        @testset "Edge Cases" begin
            f_edge = (t, y, yp) -> 0
            g_edge = (t, y) -> 0
            bc_edge = (ya, yb) -> 0
            tspan_edge = (0.0, 1e-5)
            y0_edge = [0.0]
            algebraic_vars_edge = [false]

            bvdae_edge = KSBVDAEProblem(f_edge, g_edge, bc_edge, tspan_edge, y0_edge, algebraic_vars_edge, y0_edge)
            @test bvdae_edge.tspan == (0.0, 1e-5)
            @test bvdae_edge.y0 == [0.0]
            @test bvdae_edge.algebraic_vars == [false]
        end

        @testset "Complex Cases" begin
            f_complex = (t, y, yp) -> sin(t) * yp + cos(y)
            g_complex = (t, y) -> y^3 - 1
            bc_complex = (ya, yb) -> ya^2 + yb^2 - 1
            tspan_complex = (0.0, 1e5)
            y0_complex = [1.0]
            algebraic_vars_complex = [true]

            bvdae_complex = KSBVDAEProblem(f_complex, g_complex, bc_complex, tspan_complex, y0_complex, algebraic_vars_complex, y0_complex)
            @test bvdae_complex.tspan == (0.0, 1e5)
            @test bvdae_complex.y0 == [1.0]
            @test bvdae_complex.algebraic_vars == [true]
        end
    end

    @testset "KSIDEProblem Tests" begin
        @testset "Valid Cases" begin
            f = (t, y, int) -> y + int
            K = (s, t) -> exp(-(t - s))
            tspan = (0.0, 1.0)
            y0 = [1.0]
            bc = x -> 0.0

            ide_problem = KSIDEProblem(f, K, bc, tspan, y0)

            @test ide_problem isa AbstractKSIDEProblem{Float64,1}
            @test ide_problem.tspan == (0.0, 1.0)
            @test ide_problem.initial_conditions == [1.0]
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSIDEProblem("invalid", "invalid", x -> 0.0, ("invalid", "invalid"), ["invalid"])
        end

        @testset "Edge Cases" begin
            f_edge = (t, y, int) -> 0
            K_edge = (s, t) -> 1
            bc_edge = x -> 0.0
            tspan_edge = (0.0, 0.0)
            y0_edge = [0.0]

            ide_edge = KSIDEProblem(f_edge, K_edge, bc_edge, tspan_edge, y0_edge)
            @test ide_edge.tspan == (0.0, 0.0)
            @test ide_edge.initial_conditions == [0.0]
        end

        @testset "Complex Cases" begin
            f_complex = (t, y, int) -> sin(t) * y + int
            K_complex = (s, t) -> exp(-s * t)
            bc_complex = x -> 0.0
            tspan_complex = (0.0, 3.0)
            y0_complex = [1.0]

            ide_complex = KSIDEProblem(f_complex, K_complex, bc_complex, tspan_complex, y0_complex)
            @test ide_complex.tspan == (0.0, 3.0)
            @test ide_complex.initial_conditions == [1.0]
        end
    end

    @testset "KSODEProblem Tests" begin
        @testset "Valid Cases" begin
            ode = KSODEProblem((t, y) -> -0.5 * y, (0.0, 1.0), [1.0])
            @test ode.tspan == (0.0, 1.0)
            @test ode.initial_conditions == [1.0]
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSODEProblem("invalid", (0.0, 1.0), ["invalid"])
        end

        @testset "Edge Cases" begin
            ode_edge = KSODEProblem((t, y) -> 0, (0.0, 0.0), [0.0])
            @test ode_edge.tspan == (0.0, 0.0)
            @test ode_edge.initial_conditions == [0.0]
        end

        @testset "Complex Cases" begin
            ode_complex = KSODEProblem((t, y) -> sin(t) - cos(y), (0.0, 3.14), [2.0])
            @test ode_complex.tspan == (0.0, 3.14)
            @test ode_complex.initial_conditions == [2.0]
        end
    end

    @testset "KSPDEProblem Tests" begin
        @testset "Valid Cases" begin
            @testset "Simple PDE" begin
                pde_func = (u, p, t) -> u - p[1] * t
                bc = (u, p, t) -> u - p[2]
                tspan = (0.0, 1.0)
                domain = ((0.0, 1.0), (0.0, 1.0))
                ic = [1.0, 0.0]

                pde_problem = KSPDEProblem(pde_func, bc, tspan, domain, ic)

                @test pde_problem.pde == pde_func
                @test pde_problem.boundary_conditions == bc
                @test pde_problem.tspan == tspan
                @test pde_problem.domain == domain
                @test pde_problem.initial_conditions == ic
            end

            @testset "Complex PDE" begin
                pde_func = (u, p, t) -> u + sin(p[1] * t)
                bc = (u, p, t) -> u - cos(p[2] * t)
                tspan = (0.0, 10.0)
                domain = ((0.0, 5.0), (0.0, 5.0))
                ic = [0.5, 0.1]

                pde_problem = KSPDEProblem(pde_func, bc, tspan, domain, ic)

                @test pde_problem.pde == pde_func
                @test pde_problem.boundary_conditions == bc
                @test pde_problem.tspan == tspan
                @test pde_problem.domain == domain
                @test pde_problem.initial_conditions == ic
            end
        end

        @testset "Edge Cases" begin
            @testset "Zero Time Span" begin
                pde_func = (u, p, t) -> u + p[1] * t
                bc = (u, p, t) -> u
                tspan = (0.0, 0.0)
                domain = ((0.0, 1.0), (0.0, 1.0))
                ic = [0.0]

                pde_problem = KSPDEProblem(pde_func, bc, tspan, domain, ic)

                @test pde_problem.tspan == (0.0, 0.0)
                @test pde_problem.domain == ((0.0, 1.0), (0.0, 1.0))
                @test pde_problem.initial_conditions == [0.0]
            end

            @testset "Point Domain" begin
                pde_func = (u, p, t) -> u + p[1] * t
                bc = (u, p, t) -> u
                tspan = (0.0, 1.0)
                domain = ((0.0, 0.0), (0.0, 0.0))
                ic = [0.0]

                pde_problem = KSPDEProblem(pde_func, bc, tspan, domain, ic)

                @test pde_problem.tspan == (0.0, 1.0)
                @test pde_problem.domain == ((0.0, 0.0), (0.0, 0.0))
                @test pde_problem.initial_conditions == [0.0]
            end
        end

        @testset "Invalid Cases" begin
            valid_func = (u, p, t) -> u
            valid_bc = (u, p, t) -> 0.0
            valid_tspan = (0.0, 1.0)
            valid_domain = ((0.0, 1.0), (0.0, 1.0))
            valid_ic = [1.0, 0.0]

            @test_throws MethodError KSPDEProblem("not a function", valid_bc, valid_tspan, valid_domain, valid_ic)
            @test_throws MethodError KSPDEProblem(valid_func, "not a function", valid_tspan, valid_domain, valid_ic)
            @test_throws MethodError KSPDEProblem(valid_func, valid_bc, (0.0, 1.0, 2.0), valid_domain, valid_ic)  # Invalid tspan
            @test_throws MethodError KSPDEProblem(valid_func, valid_bc, valid_tspan, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), valid_ic)  # Invalid domain
            @test_throws MethodError KSPDEProblem(valid_func, valid_bc, valid_tspan, valid_domain, "not an array")
        end
    end

    @testset "KSDAEProblem Tests" begin
        @testset "Valid Cases" begin
            dae = KSDAEProblem((t, y, yp) -> yp + y - t^2, (0.0, 1.0), [1.0])
            @test dae.tspan == (0.0, 1.0)
            @test dae.initial_conditions == [1.0]
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSDAEProblem("invalid", (0.0, 1.0), ["invalid"])
        end

        @testset "Edge Cases" begin
            dae_edge = KSDAEProblem((t, y, yp) -> 0, (0.0, 0.0), [0.0])
            @test dae_edge.tspan == (0.0, 0.0)
            @test dae_edge.initial_conditions == [0.0]
        end
    end

    @testset "KSMovingBoundaryPDEProblem Tests" begin
        @testset "Valid Case" begin
            pde_func = function (u, p, t)
                return sum(diff(diff(u, dims=1), dims=1)) - diff(u, dims=2)
            end
            bc = function (u, p, t)
                return 0.0
            end
            tspan = (0.0, 1.0)
            domain = ((0.0, 1.0), (0.0, 1.0))
            boundary_motion = function (t)
                return 1.0 + 0.1 * t
            end
            ic = [1.0, 0.0, 0.0]

            moving_boundary_pde_problem = KSMovingBoundaryPDEProblem(pde_func, bc, tspan, domain, boundary_motion, ic)

            @test moving_boundary_pde_problem.pde == pde_func
            @test moving_boundary_pde_problem.boundary_conditions == bc
            @test moving_boundary_pde_problem.tspan == tspan
            @test moving_boundary_pde_problem.domain == domain
            @test moving_boundary_pde_problem.boundary_motion == boundary_motion
            @test moving_boundary_pde_problem.initial_conditions == ic
        end

        @testset "Edge Case" begin
            pde_func = function (u, p, t)
                return zeros(size(u))  # Zero PDE
            end
            bc = function (u, p, t)
                return u  # Identity BC
            end
            tspan = (0.0, 0.0)  # Zero time span
            domain = ((0.0, 0.0), (0.0, 0.0))  # Point domain
            boundary_motion = function (t)
                return 0.0  # Stationary boundary
            end
            ic = [0.0]  # Single point IC

            moving_boundary_pde_problem = KSMovingBoundaryPDEProblem(pde_func, bc, tspan, domain, boundary_motion, ic)

            @test moving_boundary_pde_problem.tspan == (0.0, 0.0)
            @test moving_boundary_pde_problem.domain == ((0.0, 0.0), (0.0, 0.0))
            @test moving_boundary_pde_problem.initial_conditions == [0.0]
        end

        @testset "Invalid Cases" begin
            valid_func = (u, p, t) -> u
            valid_bc = (u, p, t) -> 0.0
            valid_tspan = (0.0, 1.0)
            valid_domain = ((0.0, 1.0), (0.0, 1.0))
            valid_boundary_motion = (t) -> t
            valid_ic = [1.0, 0.0, 0.0]

            @test_throws MethodError KSMovingBoundaryPDEProblem("not a function", valid_bc, valid_tspan, valid_domain, valid_boundary_motion, valid_ic)
            @test_throws MethodError KSMovingBoundaryPDEProblem(valid_func, valid_bc, valid_tspan, valid_domain, "not a function", valid_ic)
        end
    end

    @testset "KSPIDEProblem Tests" begin
        @testset "Valid Case" begin
            pide_func = function (u, p, t)
                return sum(diff(diff(u, dims=1), dims=1)) - diff(u, dims=2)
            end
            K = function (x, s)
                return exp(-abs(x - s))
            end
            bc = function (u, p, t)
                return 0.0
            end
            tspan = (0.0, 1.0)
            domain = ((0.0, 1.0), (0.0, 1.0))
            ic = [1.0, 0.0, 0.0]

            pide_problem = KSPIDEProblem(pide_func, K, bc, tspan, domain, ic)

            @test pide_problem.pide == pide_func
            @test pide_problem.K == K
            @test pide_problem.boundary_conditions == bc
            @test pide_problem.tspan == tspan
            @test pide_problem.domain == domain
            @test pide_problem.initial_conditions == ic
        end

        @testset "Edge Case" begin
            pide_func = function (u, p, t)
                return zeros(size(u))  # Zero PIDE
            end
            K = function (x, s)
                return 0.0  # Zero kernel
            end
            bc = function (u, p, t)
                return u  # Identity BC
            end
            tspan = (0.0, 0.0)  # Zero time span
            domain = ((0.0, 0.0), (0.0, 0.0))  # Point domain
            ic = [0.0]  # Single point IC

            pide_problem = KSPIDEProblem(pide_func, K, bc, tspan, domain, ic)

            @test pide_problem.tspan == (0.0, 0.0)
            @test pide_problem.domain == ((0.0, 0.0), (0.0, 0.0))
            @test pide_problem.initial_conditions == [0.0]
        end

        @testset "Complex Case" begin
            pide_func = function (u, p, t)
                return sum(diff(diff(u, dims=1), dims=1)) - diff(u, dims=2) + sin.(u) .* cos(t)
            end
            K = function (x, s)
                return 1 / (1 + (x - s)^2)  # Lorentzian kernel
            end
            bc = function (u, p, t)
                return exp(-t) * sum(u)
            end
            tspan = (0.0, Inf)  # Infinite time span
            domain = ((-Inf, Inf), (-Inf, Inf))  # Infinite domain
            ic = [0.0]  # Single point IC

            pide_problem = KSPIDEProblem(pide_func, K, bc, tspan, domain, ic)

            @test pide_problem.tspan == (0.0, Inf)
            @test pide_problem.domain == ((-Inf, Inf), (-Inf, Inf))
            @test length(pide_problem.initial_conditions) == 1
        end

        @testset "Invalid Cases" begin
            valid_func = (u, p, t) -> u
            valid_K = (x, s) -> exp(-abs(x - s))
            valid_bc = (u, p, t) -> 0.0
            valid_tspan = (0.0, 1.0)
            valid_domain = ((0.0, 1.0), (0.0, 1.0))
            valid_ic = [1.0, 0.0, 0.0]

            @test_throws MethodError KSPIDEProblem("not a function", valid_K, valid_bc, valid_tspan, valid_domain, valid_ic)
            @test_throws MethodError KSPIDEProblem(valid_func, "not a function", valid_bc, valid_tspan, valid_domain, valid_ic)
            @test_throws MethodError KSPIDEProblem(valid_func, valid_K, valid_bc, valid_tspan, valid_domain, [])
        end
    end

    @testset "KSCoupledProblem Tests" begin
        @testset "Valid Case" begin
            # Define two simple ODEs
            ode1 = function (du, u, p, t)
                du[1] = -u[1]
            end
            ode2 = function (du, u, p, t)
                du[1] = -2 * u[1]
            end

            problem1 = KSODEProblem(ode1, (0.0, 1.0), [1.0])
            problem2 = KSODEProblem(ode2, (0.0, 1.0), [2.0])

            coupling_terms = Matrix{Union{Nothing,Function}}(undef, 2, 2)
            coupling_terms[1, 1] = nothing
            coupling_terms[1, 2] = (u1, u2, p, t) -> u1[1]
            coupling_terms[2, 1] = (u2, u1, p, t) -> u2[1]
            coupling_terms[2, 2] = nothing

            coupled_problem = KSCoupledProblem{Float64}([problem1, problem2], coupling_terms)

            @test length(coupled_problem.problems) == 2
            @test coupled_problem.problems[1] == problem1
            @test coupled_problem.problems[2] == problem2
            @test coupled_problem.coupling_terms == coupling_terms
        end

        @testset "Edge Case" begin
            # Define an ODE
            ode1 = function (du, u, p, t)
                du[1] = -u[1]
            end

            problem1 = KSODEProblem(ode1, (0.0, 1.0), [1.0])

            coupling_terms = Matrix{Union{Nothing,Function}}(undef, 1, 1)
            coupling_terms[1, 1] = nothing

            coupled_problem = KSCoupledProblem{Float64}([problem1], coupling_terms)

            @test length(coupled_problem.problems) == 1
            @test coupled_problem.problems[1] == problem1
            @test coupled_problem.coupling_terms == coupling_terms
        end

        @testset "Complex Case" begin
            # Define two complex ODEs
            ode1 = function (du, u, p, t)
                du[1] = -u[1] * sin(t)
            end
            ode2 = function (du, u, p, t)
                du[1] = -2 * u[1] * cos(t)
            end

            problem1 = KSODEProblem(ode1, (0.0, Float64(π)), [1.0])
            problem2 = KSODEProblem(ode2, (0.0, Float64(π)), [2.0])

            coupling_terms = Matrix{Union{Nothing,Function}}(undef, 2, 2)
            coupling_terms[1, 1] = nothing
            coupling_terms[1, 2] = (u1, u2, p, t) -> u1[1] * u2[1]
            coupling_terms[2, 1] = (u2, u1, p, t) -> u2[1] * u1[1]
            coupling_terms[2, 2] = nothing

            coupled_problem = KSCoupledProblem{Float64}([problem1, problem2], coupling_terms)

            @test length(coupled_problem.problems) == 2
            @test coupled_problem.problems[1] == problem1
            @test coupled_problem.problems[2] == problem2
            @test coupled_problem.coupling_terms == coupling_terms
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSCoupledProblem("not a problem", "not coupling terms")
            valid_problem = KSODEProblem((du, u, p, t) -> du[1] = -u[1], (0.0, 1.0), [1.0])
            @test_throws MethodError KSCoupledProblem([valid_problem], "not coupling terms")
            valid_coupling_terms = Matrix{Union{Nothing,Function}}(undef, 1, 1)
            valid_coupling_terms[1, 1] = nothing
            @test_throws MethodError KSCoupledProblem("not a problem", valid_coupling_terms)
        end
    end

    @testset "KSDiscretizedProblem Tests" begin
        @testset "Valid Cases" begin
            f(x) = x^2
            dp = KSDiscretizedProblem([0.0, 1.0], [[0.0, 0.5, 1.0]], [1 0; 0 1.0], [1.0, 2.0], Function[])
            @test dp.time_nodes == [0.0, 1.0]
            @test dp.spatial_nodes == [[0.0, 0.5, 1.0]]
            @test dp.system_matrix == [1 0; 0 1]
            @test dp.initial_conditions == [1.0, 2.0]
            @test length(dp.problem_functions) == 0
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSDiscretizedProblem("invalid", "invalid", "invalid", "invalid", "invalid")
        end

        @testset "Edge Cases" begin
            dp_edge = KSDiscretizedProblem(Float64[], Vector{Float64}[], Matrix{Float64}(undef, 0, 0), Float64[], Function[])
            @test isempty(dp_edge.time_nodes)
            @test isempty(dp_edge.spatial_nodes)
            @test size(dp_edge.system_matrix) == (0, 0)
            @test isempty(dp_edge.initial_conditions)
            @test isempty(dp_edge.problem_functions)
        end

        @testset "Complex Cases" begin
            dp_complex = KSDiscretizedProblem([0.0, 0.5, 1.0], [[0.0, 0.25, 0.5, 0.75, 1.0]], rand(3, 3), [1.0, 2.0, 3.0], [(x) -> sin(x), (x) -> cos(x)])
            @test length(dp_complex.time_nodes) == 3
            @test length(dp_complex.spatial_nodes[1]) == 5
            @test size(dp_complex.system_matrix) == (3, 3)
            @test length(dp_complex.initial_conditions) == 3
            @test length(dp_complex.problem_functions) == 2
        end
    end

    @testset "KSTimeSteppingSolver Tests" begin
        @testset "Valid Cases" begin
            tss = KSTimeSteppingSolver(:euler, 0.1, 1.0, 1e-5)
            @test tss.method == :euler
            @test tss.dt == 0.1
            @test tss.t_final == 1.0
            @test tss.tolerance == 1e-5
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSTimeSteppingSolver(:invalid, "0.1", "1.0", "1e-5")
        end

        @testset "Edge Cases" begin
            tss_edge = KSTimeSteppingSolver(:euler, 0.0, 0.0, 0.0)
            @test tss_edge.dt == 0.0
            @test tss_edge.t_final == 0.0
            @test tss_edge.tolerance == 0.0
        end

        @testset "Complex Cases" begin
            tss_complex = KSTimeSteppingSolver(:rk4, 0.01, 10.0, 1e-10)
            @test tss_complex.method == :rk4
            @test tss_complex.dt == 0.01
            @test tss_complex.t_final == 10.0
            @test tss_complex.tolerance == 1e-10
        end
    end

    @testset "KSDirectSolver Tests" begin
        @testset "Valid Cases" begin
            ds = KSDirectSolver(:lu, 1e-5)
            @test ds.method == :lu
        end
    end

    @testset "KSIterativeSolver Tests" begin
        @testset "Valid Cases" begin
            is = KSIterativeSolver(:cg, 100, 1e-5, nothing)
            @test is.method == :cg
            @test is.max_iter == 100
            @test is.tolerance ≈ 1e-5
            @test is.preconditioner === nothing
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSIterativeSolver(:invalid, "100", "1e-5", "nothing")
        end

        @testset "Edge Cases" begin
            is_edge = KSIterativeSolver(:cg, 0, 0.0, nothing)
            @test is_edge.max_iter == 0
            @test is_edge.tolerance == 0.0
        end
    end

    @testset "KSAMGSolver Tests" begin
        @testset "Valid Cases" begin
            ams = KSAMGSolver(200, 1e-4, :jacobi)
            @test ams.max_iter == 200
            @test ams.tolerance ≈ 1e-4
            @test ams.smoother == :jacobi
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError KSAMGSolver("200", "1e-4", :invalid)
        end

        @testset "Edge Cases" begin
            ams_edge = KSAMGSolver(0, 0.0, :jacobi)
            @test ams_edge.max_iter == 0
            @test ams_edge.tolerance == 0.0
        end

        @testset "Complex Cases" begin
            ams_complex = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            @test ams_complex.max_iter == 1000
            @test ams_complex.tolerance ≈ 1e-6
            @test ams_complex.smoother == :gauss_seidel
        end
    end
end
