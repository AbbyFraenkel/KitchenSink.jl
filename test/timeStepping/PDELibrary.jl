using Test
import QuadGK: quadgk


x_values = range(-10, stop = 10, length = 5)  # 5 x values
t_values = range(0, stop = 10, length = 5)  # 5 t values
c_values = [0.01, 100]  # small and large wave speeds
α_values = [0.01, 100]  # small and large diffusivities

# Define the initial conditions
initial_conditions = [
    ("normal frequency", x -> sin(x), x -> cos(x)),
    ("high frequency", x -> sin(10x), x -> cos(10x)),
    ("sharp transitions", x -> abs(x) <= 1 ? 1 : 0, x -> abs(x) <= 1 ? 0 : 1),
    ("stiff problems", x -> x <= 0 ? sin(x) : exp(-x), x -> x <= 0 ? cos(x) : exp(-x)),
]

# Define the functions
f(x) = sin.(x)  # initial displacement
g(x) = cos(x)  # initial velocity

function integral_part(x, t_values, c, g)
    results = []
    errors = []
    for t in t_values
        for x in x_values
            result, error = quadgk(g, x - c * t, x + c * t, rtol = 1e-8)
            push!(results, result / (2 * c))
            push!(errors, error / (2 * c))
        end
    end
    return results, errors
end

# Call the function with your parameters
results, errors = integral_part(x_values, t_values, c, g)


function f_heat(x_values, t_values, α, f, g)
    f_heat_values = zeros(length(t_values), length(x_values))
    for i in 1:length(x_values)
        for j in 1:length(t_values)
            x, t = x_values[i], t_values[j]
            f_heat_values[j, i] = exp(-α^2 * t) * f(x)
        end
    end
    return f_heat_values
end

function test_condition_heat(condition, PDEsolver, x_values, t_values, α, f)
    @testset "Testing $condition" begin
        yPDE = PDE_Solver(x_values, t_values, α, f)
        yAnalytical = f_heat(x_values, t_values, α, f, 0)
        for i in 1:length(x_values)
            for j in 1:length(t_values)
                @test yPDE[j, i]≈yAnalytical[j, i] atol=1e-12
            end
        end
    end
end

@testset "Testing PDEsolver on Heat Equation" begin
    for i in eachindex(α_values)
        for j in 1:length(initial_conditions)
            α = α_values[i]
            condition, f, g = initial_conditions[j]
            test_condition_heat(condition, PDE_Solver, u_heat, x_values, t_values, α, f)
        end
    end
end


function f_laplace(x_values, y_values, f)
    f_laplace_values = zeros(length(y_values), length(x_values))
    for i in 1:length(x_values)
        for j in 1:length(y_values)
            x, y = x_values[i], y_values[j]
            f_laplace_values[j, i] = f(x) * f(y)
        end
    end
    return f_laplace_values
end

function test_condition_laplace(condition, PDEsolver, x_values, y_values, f)
    @testset "Testing Laplace equation with $condition" begin
        yPDE = PDEsolver(x_values, y_values, f)
        yAnalytical = f_laplace(x_values, y_values, f)
        for i in 1:length(x_values)
            for j in 1:length(y_values)
                @test yPDE[j, i]≈yAnalytical[j, i] atol=1e-12
            end
        end
    end
end

@testset "Testing PDEsolver on Laplace's Equation" begin
    for (condition, f) in initial_conditions
        test_condition_laplace(condition, PDE_Solver, f_laplace, x_values, 10x_values, f, g)
    end
end


function u_wave(x_values, t_values, c, f, g)
    results, errors = integral_part(x_values, t_values, c, g)
    u_wave_values = zeros(length(t_values), length(x_values))
    for i in 1:length(x_values)
        for j in 1:length(t_values)
            x, t = x_values[i], t_values[j]
            integral = results[i]
            u_wave_values[j, i] = 0.5 * (f(x - c * t) + f(x + c * t)) + integral
        end
    end
    return u_wave_values
end


function test_condition_wave(condition, PDEsolver, u_wave, x_values, t_values, c, f, g)
    @testset "Testing $condition with c = $c" begin
        yPDE = PDEsolver(x_values, t_values, c, f, g)
        yAnalytical = u_wave(x_values, t_values, c, f, g)
        for i in 1:length(x_values)
            for j in 1:length(t_values)
                @test yPDE[j, i]≈yAnalytical[j, i] atol=1e-12
            end
        end
    end
end

@testset "Testing PDEsolver on Wave Equation" begin
    for i in 1:length(c_values)
        for j in 1:length(initial_conditions)
            c = c_values[i]
            condition = initial_conditions[j]
            test_condition_wave(condition, PDE_Solver, u_wave, x_values, t_values, c, f, g)
        end
    end
end
