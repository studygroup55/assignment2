// This model defines the choices of an agent playing the Matching Pennies game. 
// The model defines an agent following the win-stay-lose-shift strategy to some extent (how much it follows the strategy the parameter to be estimated)

// Defining the data input for the model
data {
  int<lower=1> n_trials; // Number of trials, integer
  array[n_trials] int choice; // Current choice, 0 (left) and 1 (right), integer
  vector<lower=-1, upper=1>[n_trials] strategy_choice; // Whether to choose left (-1) or right (1) depending on the previous outcome and the WSLS strategy
}
  
// Defining the parameter to be estimated
parameters {
  real<lower=0, upper=1> ruleFollowing; // How much the agent is following the WSLS strategy (probability, from 0 to 1)
}

// Transforming parameters
transformed parameters {
  vector[n_trials] rate; // Defining the rate which informs choice
  rate = strategy_choice*ruleFollowing; // Rate is the multiple of the next choice according to strategy and how probably the agent is to follow the strategy
}

// Defining the priors and model
model {
  target += normal_lpdf(ruleFollowing | 0, 1); // Defining a normal distribution with mean 0 and SD 1 for the rule-following parameter 
  target += bernoulli_logit_lpmf(choice | rate); // Defining the model: The binary choice (i.e. a Bernoulli distribution) as predicted by rate (assumed to be in a logit space)
}
 
// Transforms log odds into probabilities 
generated quantities {
  vector[n_trials] rate_p; // Defining a new vector of the rate transformed into probability
  rate_p = inv_logit(rate); // Transforming rate into probability (0 to 1) by a sigmoid transformation
}


