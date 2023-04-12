//
// This Stan program defines the memory model - The second model of our assignment 2 in ACM
// Right now, we have included a "learning" parameter in the model specification, but not elsewhere. Look into this


// The input (data) for the model. n of trials and h for (right and left) hand

data {
  int<lower=1> n_trials; // Number of trials
  array[n_trials] int choice; // choice of self
  array[n_trials] int other; // Choice of other
}

 
// The parameters accepted by the model.
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
  real<lower=0, upper=1> alpha; //EMA smoothing factor
}

 

// The model to be estimated.
model {
  vector[n_trials] memory; //Vector of n_trials for storing memory 
  // Priors
  target += beta_lpdf(alpha | 1, 1); //prior for alpha made from beta lpdf with alpha = 1 and beta = 1
  target += normal_lpdf(bias | 0, .3); //prior for bias made from normal lpdf with mean = 0 and sd = 0.3 
  target += normal_lpdf(beta | 0, 1); //prior for beta made from normal lpdf with mean = 0 and sd = 1

  // Model, looping to keep track of memory
  for (trial in 1:n_trials) {  //for-loop with n_trials iterations
    if (trial == 1) { // if it is the first trial we set memory to 0.5 (random)
      memory[trial] = 0.5;
    }
    target += bernoulli_logit_lpmf(choice[trial] | bias + beta * memory[trial]); // choice is distributed according to a bernoulli. The theta of the bernoulli is based on bias + beta *memory
    
    if (trial < n_trials){
      memory[trial + 1] = alpha * memory[trial] + (1-alpha) * other[trial]; # Exponential moving average
    }
  }
}

generated quantities { 
  real bias_prior; //Creating real values for priors and posteriors for each parameter
  real beta_prior;
  real alpha_prior;
  real bias_posterior;
  real beta_posterior;
  real alpha_posterior;
  int<lower=0, upper=n_trials> bias_prior_preds;  //Creating int values for prior and posterior predictions
  int<lower=0, upper=n_trials> beta_prior_preds;
  int<lower=0, upper=n_trials> alpha_prior_preds;
  int<lower=0, upper=n_trials> bias_posterior_preds;
  int<lower=0, upper=n_trials> beta_posterior_preds;
  int<lower=0, upper=n_trials> alpha_posterior_preds;
  
  bias_prior = inv_logit(normal_rng(0, .3)); #Defining the priors
  beta_prior = inv_logit(normal_rng(0, 1));
  alpha_prior = inv_logit(beta_rng(1, 1));
  
  bias_posterior = inv_logit(bias); #transforming of our posteriors to probability space
  beta_posterior = inv_logit(beta);
  alpha_posterior = alpha;
  
  bias_prior_preds = binomial_rng(n_trials, bias_prior); //Creating predicitons based on prior
  beta_prior_preds = binomial_rng(n_trials, beta_prior);
  alpha_prior_preds = binomial_rng(n_trials, alpha_prior);
  
  bias_posterior_preds = binomial_rng(n_trials, bias_posterior); //Creating predicitons based on posterior
  beta_posterior_preds = binomial_rng(n_trials, beta_posterior);
  alpha_posterior_preds = binomial_rng(n_trials, alpha_posterior);
  
}

 
