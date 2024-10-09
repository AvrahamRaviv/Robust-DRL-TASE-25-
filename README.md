# Robust Deep Reinforcement Learning Agents Using Formal Verification

In the Frozen Lake environment, the goal is to move an agent to a specific coordinate on a grid while avoiding obstacles, referred to as ”holes.” The agent can take steps in four directions: north, south, east, or west. To increase complexity, the agent must learn to navigate from any possible starting position on the board to the goal, as there is no fixed starting state. Each environment initialization starts the agent in a new random position that is not occupied by a hole or the goal. The success rate of the trained network is measured by calculating the percentage of states from which the agent successfully navigates to the goal, following the policy learned by the network.

### Requirements
maraboupy 2.0.0 <br/>
matplotlib 3.9.2 <br/>
numpy 2.1.1 <br/>
onnx 1.16.2 <br/>
torch 2.4.1 

### Main Files
_verify_deep_q_learning_with_replay_with_reward_shaping.py_ - standard DQN with replay buffer and reward shaping. <br/><br/>
_verify_double_deep_v1.py_ - the original Double Deep Q-Learning Algorithm. <br/><br/>
_verify_double_deep_v2.py_ - the revised Double Deep Q-learning Algorithm. <br/><br/>
_check_verification_group_1_in_model.py, check_verification_group_2_in_model.py_ - responsible for verifying properties in the model. <br/><br/>
_Frozen_Lake_Environment.py_ - implementation of the environment.<br/><br/>
_main.py_ - main file.

### Running the code
To compare the preformance of the developed verification-based backpropagation algorithm to the standard algorithms, follow these steps: <br/>
1) Pick the wanted algorithm by placing its name in the "runs" list in line  of main.py. <br/>
2) At main.py, add the wanted layout of your choice to the environment, and set the size of your board. The method _train_ recives as a parameter the maximum amount of episodes. Set the parameter max_time in the chosen algorithm's file to the number of seconds you wish the algorithm will run. <br/>
3) Run main.py

### Results
![image](https://github.com/user-attachments/assets/c70563c8-bb2a-48c5-a80f-1c246ddb806d) <br/>
Three trials were conducted for each combination of algorithm and layout, depicted above, to account for the randomness inherent in the algorithms (the networks' parameters, epsilon greeedy strategy, replay memory sampling, etc.). The averages of these trials are detailed below: <br/>
![image](https://github.com/user-attachments/assets/0a946584-23c2-4f01-9702-2352c09e8a61) <br/>
It is recommended to run several trials, to account for the randomness inherent in the algorithms. 
