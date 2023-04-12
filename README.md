This repo contains the code and models used to model the Matching Pennies Task with two different cognitive models: One following a win-stay-lose-shift strategy with some probability (rulefollowing agent), and one using a continuously updated memory to inform the decision. 

The file rulefollowing_model.Rmd contain the code for the first agent. This code utilizes both the Stan files rulefollowing_agent.stan and rulefollowing_agent_test.stan (which is identical to the other stan file but enables prior sensitivity checks). 

The file memory_model.Rmd contain the code for the second agent. This code utilizes both the Stan files memory_agent.stan and memory_agent_test.stan (which is identical to the other stan file but enables prior sensitivity checks).
