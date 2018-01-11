import numpy as np
import math
import sys
import scipy.integrate as integrate


class ProbabilityModel:

	# Returns a single sample (independent of values returned on previous calls).
	# The returned value is an element of the model's sample space.
	def sample(self):
		pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
	
	# Initializes a univariate normal probability model object
	# parameterized by mu and (a positive) sigma
	def __init__(self,mu,sigma):
		self.mu = mu
		self.sigma = sigma

	# generate a random variable within the provided signma and mu
	def next(self):
		x = np.random.uniform()
		error_thresh = 0.0001
		error = np.inf
		ub, lb, mid = self.mu+1000*self.sigma, self.mu-1000*self.sigma, self.mu
		prob_l, prob_u, prob_mid = 0.0,1.0,0.5
		i = 0
		while error > error_thresh and i < 100:
			if prob_mid < x:                                     #move to larger range
				lb = mid
				prob_l = self.cumulative_density_function(lb)
			else:                                                #move to smaller range
				ub = mid
				prib_u = self.cumulative_density_function(ub)
			mid = (ub+lb)/2.0
			prob_mid = self.cumulative_density_function(mid)
			error = abs(prob_mid-x)
			i += 1
		print x
		return mid

	def cumulative_density_function(self, x):
		return 0.5 + 0.5 * math.erf((x-self.mu)/(self.sigma*math.sqrt(2)))

	def density_function(self, x):
		return 1/(self.sigma * (2*math.pi)**(1/2)) * math.exp(-(x-self.mu)**2/2*self.sigma**2)

# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
	
	# Initializes a multivariate normal probability model object 
	# parameterized by Mu (numpy.array of size D x 1) expectation vector 
	# and symmetric positive definite covariance Sigma (numpy.array of size D x D)
	def __init__(self,Mu,Sigma):
		self.Mu = Mu
		self.Sigma = Sigma

	# generate a vector of normally distributed variables
	def next(self):
		X = [np.random.uniform() for i in range(len(Mu))]

	

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
	
	# Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
	# probability model object with distribution parameterized by the atomic probabilities vector
	# ap (numpy.array of size k).
	def __init__(self,ap):
		pass


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
	
	# Initializes a mixture-model object parameterized by the
	# atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
	# probability models pm
	def __init__(self,ap,pm):
		pass


model = UnivariateNormal(100,10)
print model.next()