

analytical_ode(t) = exp(-t)

    @testset "ODEs" begin
        @testset "KSODEProblem" begin
            # Test ODE: dy/dt = -y, y(0) = 1
            ode = KSODEProblem(
                (t, y) -> -y,
                (0.0, 1.0),
                [1.0]
            )
            @test ode isa KSODEProblem
            @test ode.tspan == (0.0, 1.0)
            @test ode.initial_conditions == [1.0]


        end
    end
