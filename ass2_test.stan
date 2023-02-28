stan_model = "
// This STAN model infers a specific bias from a sequence of 0s and 1s (choices in the matching pennies game - with a WSLS strategy with noise implemented)
//


// The input (data) for the model 
data {
  int<lower=1> n; //number of trials. can't be lower than 1
  array[n] int choice; //array of choices of length n, for player
}


// The parameters accepted by the model
parameters{
  real<lower=0, upper=1> noise; //the bias (theta) of a person is a number between 0 and 1
}

// The model to be estimated
model{
  target += normal(bias | 0, 1); // The prior for our bias is a uniform distribution between 0 and 1
  target += bernoulli_logit_lpmf(player | bias);
}
"

write_stan_file(
  stan_model,
  dir = "./ass2_test.stan",
  basename = "W3_SimpleBernoulli.stan")