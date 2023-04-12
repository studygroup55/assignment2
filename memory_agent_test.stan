// The input (data) for the model. n of trials and h for (right and left) hand

data {
  int<lower=1> n_trials; // Number of trials
  array[n_trials] int choice; // choice of self
  array[n_trials] int other; // Choice of other
  real prior_mean_bias; //prior mean for bias
  real<lower=0> prior_sd_bias; //prior standard deviation for bias
  real prior_mean_beta; //prior mean for beta
  real<lower=0> prior_sd_beta; //prior standard deviation for beta
}

 
// The parameters accepted by the model.
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
  real<lower=0, upper=1> alpha; // EMA smoothing factor
}

 

// The model to be estimated.
model {
  vector[n_trials] memory;  //Vector of n_trials for storing memory 
  // Priors
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias); //prior for bias with varying means and standard deviations
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta); //prior for beta with varying means and standard deviations
  target += beta_lpdf(alpha | 1, 1); //prior for alpha made from beta lpdf with alpha = 1 and beta = 1

  // Model, looping to keep track of memory
  for (trial in 1:n_trials) {  //for-loop with n_trials iterations
    if (trial == 1) { // if it is the first trial we set memory to 0.5 (random)
      memory[trial] = 0.5;
    }
    target += bernoulli_logit_lpmf(choice[trial] | bias + beta * memory[trial]); // choice is distributed according to a bernoulli. The theta of the bernoulli is based on bias + beta *memory
    
    if (trial < n_trials){
      memory[trial + 1] = alpha * memory[trial] + (1-alpha) * other[trial]; // Exponential moving average
    }
  }
}

// Here we include prior posterior predictive checks

generated quantities {
  real bias_prior; //Creating real values for priors and posteriors for each parameter
  real beta_prior;
  real alpha_prior;
  real bias_posterior;
  real beta_posterior;
  real alpha_posterior;
  int<lower=0, upper=n_trials> bias_prior_preds; //Creating int values for prior and posterior predictions
  int<lower=0, upper=n_trials> beta_prior_preds;
  int<lower=0, upper=n_trials> alpha_prior_preds;
  int<lower=0, upper=n_trials> bias_posterior_preds; 
  int<lower=0, upper=n_trials> beta_posterior_preds;
  int<lower=0, upper=n_trials> alpha_posterior_preds;
  
  // Simulating prior distributions in probability space
  bias_prior = inv_logit(normal_rng(prior_mean_bias, prior_sd_bias)); //Varying priors for beta and bias
  beta_prior = inv_logit(normal_rng(prior_mean_beta, prior_sd_beta));
  alpha_prior = inv_logit(beta_rng(1, 1)); //Specifying the priors for alpha (these are constant)
  
  // Converting our estimates from log-odds to probability
  bias_posterior = inv_logit(bias); 
  beta_posterior = inv_logit(beta);
  alpha_posterior = inv_logit(alpha);
  
  // Simulating the outcome based on our prior distributions
  bias_prior_preds = binomial_rng(n_trials, bias_prior);
  beta_prior_preds = binomial_rng(n_trials, beta_prior);
  alpha_prior_preds = binomial_rng(n_trials, alpha_prior);
  
  // Simulating the outcome based on our posterior distributions
  bias_posterior_preds = binomial_rng(n_trials, bias_posterior);
  beta_posterior_preds = binomial_rng(n_trials, beta_posterior);
  alpha_posterior_preds = binomial_rng(n_trials, alpha_posterior);
  
}

