# rl_idea
idea for a new kind of reinforcement learning?

The idea is as follows:

- Create a world model that maps changeable observations to a world state
- Train world model on random samples from the world to learn a distribution of world states over observation space
- Back-propagate the changeable observations to try to make the internal world model match the state you wish to reach
  - Mirroring this in the external world should lead to the agent making decisions that lead it towards a goal state
  
Future Developments:
- More complex scenarios (images as world state, or any other kind of high dimensional output instead of just classes)
- Somehow incorporating real world observations into the internal world model without needing to back propagate those to reach a goal.
  -i.e a model cannot change what it sees to reach a desired world state, it must move itself in order to cause itself to see what it wants to see
- Incorporating sequential actions to reach a desired state (i.e, must pass through the red, then the blue, circles to reach a desired state)
  -Most likely will require memory (LSTM or RNN)

