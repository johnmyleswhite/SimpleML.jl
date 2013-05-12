function huber(r::Real, delta::Real = 1.0)
	if abs(r) <= delta
		return r^2 / 2
	else
		return delta * abs(r) - delta^2 / 2
	end
end

function dhuber(r::Real, delta::Real = 1.0)
	if abs(r) <= delta
		return r
	else
		if r <= delta
			return -1
		else
			return 1
		end
	end
end

n = 1000
p = 2
X = randn(p, n)
beta = randn(p)
y = Array(Float64, n)
for i in 1:n
	y[i] = dot(X[:, i], beta) + randn()
	if i == 99
		y[i] += 100.0
	end
end

function f(betahat::Vector)
	l = 0.0
	for i in 1:n
		l += huber(y[i] - dot(X[:, i], betahat), 1.0)
	end
	return l
end

function g!(betahat::Vector, storage::Vector)
	fill!(storage, 0.0)
	for i in 1:n
		dh = dhuber(y[i] - dot(X[:, i], betahat), 1.0)
		for j in 1:p
			storage[j] += -dh * X[j, i]
		end
	end
	return
end

using Optim

optimize(f, g!, [0., 0.], method = :l_bfgs)

using GLM

df = DataFrame()

df["X1"] = vec(X[1, :])
df["X2"] = vec(X[2, :])
df["Y"] = y

lm(:(Y ~ X1 + X2), df) # How to remove intercept?
