include("logistic_utils.jl")

srand(1)

p = 2
n = 100_000

beta, X, Y = generate_data(p, n)

writecsv("logistic.csv", hcat(X', Y))

f = generate_f(X, Y)
g! = generate_g!(X, Y)
h! = generate_h!(X, Y)

gr = Array(Float64, p)
hs = Array(Float64, p, p)

f(beta)
g!(beta, gr)
h!(beta, hs)

optimize(f, zeros(p), method = :nelder_mead)
optimize(f, zeros(p), method = :simulated_annealing)
optimize(f, g!, zeros(p), method = :l_bfgs)
optimize(f, g!, zeros(p), method = :bfgs)
optimize(f, g!, h!, zeros(p), method = :newton)

beta_hat = optimize(f, g!, zeros(p), method = :l_bfgs).minimum

f(beta_hat)
g!(beta_hat, gr)
h!(beta_hat, hs)

deviance = f(beta_hat) * 2
aic = deviance + 2 * p
