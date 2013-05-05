using Stats
using Distributions
using Optim

function generate_data(p::Integer, n::Integer)
	beta = randn(p)
	X = randn(p, n)
	Y = Array(Float64, n)
	for i in 1:n
		Y[i] = rand(Bernoulli(invlogit(dot(X[:, i], beta))))
	end
	return beta, X, Y
end

function generate_f(X, Y)
	n = size(X, 2)
	function f(coef::Vector)
		ll = 0.0
		for i in 1:n
			p = invlogit(dot(X[:, i], coef))
			ll += Y[i] * log(p) + (1.0 - Y[i]) * log(1.0 - p)
		end
		return -ll
	end
	return f
end

function generate_g!(X, Y)
	pr, n = size(X)
	function g!(coef::Vector, storage::Vector)
		fill!(storage, 0.0)
		for i in 1:n
			p = invlogit(dot(X[:, i], coef))
			for j in 1:pr
				storage[j] += -(Y[i] - p) * X[j, i]
			end
		end
	end
	return g!
end

function generate_h!(X, Y)
	pr, n = size(X)
	function h!(coef::Vector, storage::Matrix)
		fill!(storage, 0.0)
		for i in 1:n
			p = invlogit(dot(X[:, i], coef))
			for j in 1:pr
				for k in j:pr
					entry = X[j, i] * X[k, i] * exp(-dot(X[:, i], coef)) * p * p
					storage[j, k] += entry
					storage[k, j] += entry
				end
			end
		end
	end
	return h!
end
