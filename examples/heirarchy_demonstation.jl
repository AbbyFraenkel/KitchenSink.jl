using FastGaussQuadrature, Plots

X = []
W = []

for i in 1:20
    # if i %  == 0
        x, w = gausslegendre(i)
    x =  (x .+ 1) / 2
        push!(X, x)
        push!(W, w)
    end
# end

plot(X, W, title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")

scatter(X, X, title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
X
X[1]
X[2]
X[3]
X[4]
X[5]


scatter(X[1], W[1], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[2], W[2], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[3], W[3], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[4], W[4], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[5], W[5], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[6], W[6], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[7], W[7], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[8], W[8], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[9], W[9], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")


scatter(X[1], X[1], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[2], X[2], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[3], X[3], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[4], X[4], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[5], X[5], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[6], X[6], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[7], X[7], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[8], X[8], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[9], X[9], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[], X[], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[11], X[11], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[12], X[12], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[13], X[13], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[14], X[14], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[15], X[15], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[16], X[16], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter!(X[17], X[17], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")
scatter(X[18], X[18], title="Gauss-Legendre Quadrature", xlabel="x", ylabel="w")



X[]
