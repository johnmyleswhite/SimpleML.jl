include("linear_utils.jl")

srand(1)

p = 2
n = 100_000

beta, X, Y = generate_data(p, n)

writecsv("linear.csv", hcat(X', Y))

f = generate_f(X, Y)
g! = generate_g!(X, Y)
h! = generate_h!(X, Y)

gr = Array(Float64, p)
hs = Array(Float64, p, p)

f(beta)
g!(beta, gr)
h!(beta, hs)

optimize(f, [0.0, 0.0], method = :nelder_mead)
optimize(f, [0.0, 0.0], method = :simulated_annealing)
optimize(f, g!, [0.0, 0.0], method = :l_bfgs)
optimize(f, g!, [0.0, 0.0], method = :bfgs)
optimize(f, g!, h!, [0.0, 0.0], method = :newton)

beta_hat = optimize(f, g!, [0.0, 0.0], method = :l_bfgs).minimum

f(beta_hat)
g!(beta_hat, gr)
h!(beta_hat, hs)

r_squared = 1.0 - f(beta_hat) / var(Y)
