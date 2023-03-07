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
  vector<lower=-1, upper=1>[n_trials] next_choice; // based on other's actions, 1s for choosing right next time and -1 for choosing left next time
  // array[n_trials] int right_choice; // weighting of choices in the WSLS strategy, -1 being left choices and 1 being right choices, integer
}
  
parameters {
  real<lower=0, upper=1> noise; // the amount of noise altering the strategy, bound from 0 to 1 (probability?)
}

#NB: right_choice = next_choice. needs to be -1 and 1 to not cancel out. 

transformed parameters {
  vector[n_trials] rate_w_noise; // the parameter of the rate and noise
  
  //if (noise == 1) {rate_w_noise = 0.5;}
  rate_w_noise = next_choice*(1-noise); //What was the point again with 1/noise, would this be okay in our case?
  
  //rate_w_noise = next_choice*(1-noise);
  //rate_w_noise = logit(rate_w_noise);

  //if (noise == 0) {rate_w_noise = (rate + next_choice)/2;}
  //else {rate_w_noise = (rate + next_choice*(1-noise))/2;}
}

model {
  target += normal_lpdf(noise | 0, 1); // normally distributed noise
  target += bernoulli_logit_lpmf(choice | rate_w_noise); // because logit is further up (when defining rate_w_noise)
  //target += bernoulli_logit_lpmf(choice | rate_w_noise);
}
 
// This chunk transforms log odds into probabilities 
generated quantities {
  vector[n_trials] rate_w_noise_p;
  real noise_p;
  rate_w_noise_p = inv_logit(rate_w_noise);
  noise_p = inv_logit(noise);
}


