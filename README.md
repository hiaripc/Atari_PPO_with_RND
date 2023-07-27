# Playing Atari with Proximal Policy Optimization and Random Network Distillation

*WIP* 
**Consult the report.pdf file for details**

A representation of Random Network Distillation exploration bonus:
![RND](https://github.com/hiaripc/Atari_PPO_with_RND/blob/main/AAS.jpg)

## Implementation

We implemented the PPO algorithm to create an agent capable of playing the Atari game Freeway, following the paper's hyperparameters for Atari games. The implementation involves 8 parallel actors playing for 128 timesteps, for a total of 10 million steps in the environment. The network architecture is the same used in the PPO paper, described in [Nature paper on deep reinforcement learning](citet*{nature-dl}), with one head for the actor and one for the critic.

The implementation is done using the Open AI Gym library, maintained by the Farama Foundation, and EnvPool. The Gym library provides a collection of environments, ranging from simple controls to Atari games, allowing easy algorithm testing through a simple API. Moreover, it offers various Environment Wrappers that are used in this implementation in order to resize, stack and skip frames as the original work suggests.
EnvPool is a C++ library compatible with Open AI Gym APIs, which offers a batched environment pool with great performance. We used it to manage the 8 parallel actors.

It is important to note that different random seeds can have a significant influence on the results, as possible to see in the comparison between seeds in the PPO paper, thus we will conduct the experiments using 3 different seeds. Moreover, the total number of iteration steps is consistent with the original work, where 10 thousand steps are equivalent to 40 million frames. This calculation takes into account the 4 stacked frames for 128 timesteps in the environment for each PPO step, performed by 8 parallel agents.

We chose to use the game Freeway as an example: a simple Atari game where the player assumes the role of a chicken attempting to cross a street where cars are passing. Each time the player gets hit by a car, it falls back. Crossing the street gives one point, and there is a time limit.

## Results

We tested the implementation using 3 seeds, obtaining different outcomes.
With the best seed, as possible to see in [Figure Freeway PPO](/path/to/figure), after a brief period, the network starts to gain more and more rewards, reaching an impressive score of 30 after only 2000 steps. Then, the performance slightly improves, eventually reaching an impressive 34\[^1^\] as the final mean score, obtaining a state-of-the-art performance. This result is consistent and slightly better compared to the original work, where it's reported 32 as the final mean reward. On the other hand, the other two seeds, once reaching a score of 23, stop improving till the end of the training. 
In the Git-hub repository, it is possible to find a [video](https://github.com/hiaripc/Atari_PPO_with_RND/tree/main/videos/Freeway-v5) of the best agent playing.

[Figure Freeway PPO]: /path/to/figure

### Random Network Distillation

The work "Exploration by Random Network Distillation" introduced an exploration bonus using two additional networks to detect novelty in the states encountered by the agent. The exploration bonus is summed to the environment reward, producing a total reward at time t equal to r_t = e_t + i_t, where e_t is the reward given by the environment - the extrinsic reward - and i_t is the exploration bonus - the intrinsic reward.

#### The intrinsic reward

The generation of the intrinsic reward involves two neural networks: the target and the predictor, and it is based on the distillation process, which involves distilling a random neural network into a trained one. The target network is randomly initialized and sets the so-called prediction problem: given the same input to both the networks, the predictor must "guess" the output of the target network.
More precisely, given the observation produced by the target network f: O → R^k and the one produced by the predictor f̂: O → R^k, the predictor is trained by minimizing the mean squared error (MSE) loss between f and f̂. 
In this way, states dissimilar from the previous ones - produced by exploration - will result in a higher MSE, detecting novelty. 
The original work doesn't provide details about the output of the two networks, so we used the dimension of the action's space, as it is for the actor, and the calculation of the reward will follow the example given in an article on Data Iku[^2^]. Thus, the intrinsic reward i_t at a certain time t is equal to the normalized MSE using the Euclidean Norm, based on the observation of the state s_{t+1}:
\[i_t = ||\hat{f}(s_{t+1}) - f(s_{t+1})||^2_2.\]

#### Episodic and non-episodic returns

The paper argues that the exploration works better if the intrinsic rewards are calculated as non-episodic, claiming that this is actually how humans explore games. In fact, it is common that exploration leads to dangerous situations that could result in a game over. For instance, consider a scenario of a player reaching a cliff where there may be the possibility of jumping to the other side, where there could be rewards. However, the jump is difficult, and failing it would end the game. If the returns are episodic, the agent would avoid this dangerous situation that could end the episode, on the contrary, a human would repeat the episode multiple times until reaching the other side, if possible.
The non-episodic intrinsic returns R_I and the episodic extrinsic returns R_E are combined together as a sum, having R = R_E + R_I and are used to fit two value heads (critics) V_E and V_I.


