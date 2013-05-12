using Stats
using Distributions
using Optim

function generate_data(p::Integer, n::Integer)
	beta = randn(p)
	X = randn(p, n)
	Y = Array(Float64, n)
	for i in 1:n
		Y[i] = dot(X[:, i], beta) + randn()
	end
	return beta, X, Y
end

function generate_f(X::Matrix, Y::Vector, lambda::Real)
	p, n = size(X)
	mu = mean(Y)
	function f(beta::Vector)
		s = 0.0
		for i in 1:n
			s += (Y[i] - mu - dot(X[:, i], beta))^2
		end
		for j in 1:p
			s += lambda * beta[j]^2
		end
		return s
	end
	return f
end

function generate_g!(X::Matrix, Y::Vector, lambda::Real)
	p, n = size(X)
	mu = mean(Y)
	function g!(beta::Vector, storage::Vector)
		fill!(storage, 0.0)
		for i in 1:n
			epsilon = Y[i] - mu - dot(X[:, i], beta)
			for j in 1:p
				storage[j] += -2.0 * epsilon * X[j, i]
			end
		end
		for j in 1:p
			storage[j] += 2.0 * lambda * beta[j]
		end
	end
	return g!
end

function generate_h!(X::Matrix, Y::Vector, lambda::Real)
	p, n = size(X)
	function h!(beta::Vector, storage::Matrix)
		fill!(storage, 0.0)
		for j in 1:p
			for k in j:p
				entry = 2.0 * dot(vec(X[j, :]), vec(X[k, :]))
				storage[j, k] += entry
				storage[k, j] += entry
			end
		end
		for j in 1:p
			storage[j, j] += 2.0 * lambda
		end
	end
	return h!
end
