//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int t; //t the number of trials
  array[t] int c; //A one dimension array of t trials, containing int's C (that is choices)
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=0,upper=1> noise ; //adding a noise parameter, bounded between 0 and 1
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += beta_lpdf(noise|1,1); //Prior. BIAS ~ beta(1,1). LPDF = log probability density function. TARGET is a function
  target += bernoulli_lpmf(c|noise); //Likelyhood function. LPMF = log probability mass function, mass when the outcome is discrete, density is when it is continous 
}

