// Test model based loosely on this model: https://github.com/frisenborg/ACM-S22-A2-ParameterRecovery/blob/main/stan_models/WSLS.stan
// Look into Riccardos code as well about noise, maybe also look for the relevant slides: https://github.com/evasahlholdt/AdvancedCognitiveModeling2023/blob/main/stan/W3_InternalMemory.stan
// Jesper's code might also be helpful: https://github.com/JesperFischer/Advanced-cognitive-modeling/blob/main/assignment2/stan_models/rw_win_lose_vs_rw.stan
// Issue right now is in the model formulation under transformed parameters. Right_choice are integers and noise is a real number, thus can't be multiplied as I
// am trying in the current syntax. However this looks similar to what they did in the linked repository. NB: The chains can't be retrieved if we change right_choice
// into a non-integer vector
// To do: 
// Figure out how this can be specified correctly. And if this model even makes sense or maybe can be made simpler/better (e.g. with a loop)
// Should choices be coded as -1 (left) and 1 (right) as well, instead of 0 (left) and 1 (right)?

data {
  int<lower=1> n_trials; // number of trials, integer
  array[n_trials] int choice; // choices, 0 (left) and 1 (right), integer
  vector<lower=-1, upper=1>[n_trials] strategy_choice; // based on WSLS, 1 for choosing right next time and -1 for choosing left next time
}
  
parameters {
  real<lower=0, upper=1> ruleFollowing; // the amount of noise altering the strategy, bound from 0 to 1
}


transformed parameters {
  vector[n_trials] rate; // the parameter of the rate and noise, should this be a real number instead?
  //real rate_w_noise; //doesn't work
  rate = strategy_choice*ruleFollowing;
}

model {
  target += normal_lpdf(ruleFollowing | 0, 1); // normally distributed noise
  target += bernoulli_logit_lpmf(choice | rate); // because logit is further up (when defining rate_w_noise)
}
 
// This chunk transforms log odds into probabilities 
generated quantities {
  vector[n_trials] rate_p;
  //real ruleFollowing_p;
  rate_p = inv_logit(rate);
  //ruleFollowing_p = inv_logit(ruleFollowing);
}


