# rl_idea
messing around with an idea for high-dimensional space navigation.

The idea is as follows:

- Create a world model that maps changeable observations to a world state (i.e the agents x-y coordinates that can be changed through action (movement))
- Train world model on random samples from the world to learn a distribution of world states over observation space
- Back-propagate through the world model w.r.t the inputs to try to make the state output match the state you wish to reach
  - Mirroring this in the external world should lead to the agent making decisions that lead it towards a goal state
  
Future Developments:
- More complex scenarios (images as world state, or any other kind of high dimensional output instead of just classes)
- Somehow incorporating observations into the internal world model without needing to back propagate those to reach a goal.
  -i.e a model cannot change what it sees to reach a desired world state, it must move itself in order to cause itself to see what it wants to see
- Incorporating sequential actions to reach a desired state (i.e, must pass through the red, then the blue, circles to reach a desired state)
  -Most likely will require memory of some kind (LSTM or RNN, or some kind of odometry)

