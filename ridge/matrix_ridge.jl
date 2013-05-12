n = 1000
p = 2
X = randn(p, n)
beta = randn(p)
y = Array(Float64, n)
for i in 1:n
	y[i] = dot(X[:, i], beta) + randn()
end

function ridge(X::Matrix, y::Vector, lambda::Real = 0.0)
	C = X * X'
	for i in 1:p
		C[i, i] += lambda
	end
    beta = C \ X * y
end

# Simplest ridge solver
