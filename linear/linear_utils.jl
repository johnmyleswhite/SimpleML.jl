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

function generate_f(X::Matrix, Y::Vector)
	p, n = size(X)
	function f(beta::Vector)
		s = 0.0
		for i in 1:n
			s += (Y[i] - dot(X[:, i], beta))^2
		end
		return s / n
	end
	return f
end

function generate_g!(X::Matrix, Y::Vector)
	p, n = size(X)
	function g!(beta::Vector, storage::Vector)
		fill!(storage, 0.0)
		for i in 1:n
			epsilon = Y[i] - dot(X[:, i], beta)
			for j in 1:p
				storage[j] += -2.0 * epsilon * X[j, i]
			end
		end
	end
	return g!
end

function generate_h!(X::Matrix, Y::Vector)
	p, n = size(X)
	function h!(beta::Vector, storage::Matrix)
		fill!(storage, 0.0)
		for j in 1:p
			for k in j:p
				entry = 2.0 * dot(X[:, j], X[:, k])
				storage[j, k] += entry
				storage[k, j] += entry
			end
		end
	end
	return h!
end
