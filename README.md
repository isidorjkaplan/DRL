## Welcome to DRL git repo!
Thanks for visiting my DRL framework on GitHub. This is a framework that I designed to implement simple discrete RL algorythems. 

This framework supports A2C (Actor-Critic), DQN (Duelling), REINFORCE, and CEM (Cross-Entropy-Method)

It also supports a range of neural network types including Convolutional, LSTM, and Linear (FCC). 

## configurations
### How does training / ensemble work
At each step there will be a list of agents that can be used and then an ensemble that selects which agent to use at that step. In the simplest case of training only one agent we list just that agent in the `data/agents.txt` file and then in the `data/ensemble.txt` file we say we are selecting agents randomly. In this case we randomly select out of only one agent and that agent will be selected every time. 

Some more complicated configurations might involve randomly selecting between random and an agent or even between multiple different agents for an ensemble. In this case `agents.txt` will have two lines; one listing the random and one listing the DRL agent configurations. If we want to randomly select between random and agent then the `ensemble.txt` will just be a random agent. We can also use more complciated rules to decide which agent to use which can include an epsilon-greedy aproach or using another DRL agent to select. 

### Training configuration files
Each line of both files specifies arguments for training. In the example regiment below at each step it will randomly select either a random agent, a CEM model, a DQN model. Both the value and policy models will be trained at the same time. If on the other hand you ommit the 'train' flag it will use a pretrained model and will not continue to update it. 

### Configuration options
**agnet(str)**: First argument specifies the type of agent. Your options are `[RandomAgent, FixedAgent, EpsilonGreedy, ValueAgent, PolicyAgent]`
#### Options for deep agents
* **net(str)**: Specifies the type of network. Options are `[Linear, LSTM, ConvNet]`
* **model(str)**: Specifies the name of the model file. The folder it is in will be `data/models/<args.model>`
* **train(str)**: Specifies the training regiment. Leave blank for a pretrained model. Options `[cem, a2c, dqn]`
* **resume(str)**: If this flag is enabled then it will resume training the model from the iter specified in the argument
* **save_every(str)**: This flag will specify how many itterations to take backups of the model. Leave blank for no backups. 
* **epsilon(float)**: Set the probability that an agent action is replaced with random, this is to ensure sufficient exploration and should only be used in the ensemble agent. It will not print statistics. 
#### Options for FixedAgent
* **fixed_action(str)**: The numarical ID of the action that it executes each turn (discrete)
#### Options for EpsilonGreedy
Epsilon greedy should be used as the ensemble and it selects between the two agents specified in agents.txt with an epsilon-greedy stratagey.
* **epsilon_start(str)**: Starting value for epsilon
* **epsilon_finish(str)**: Final value for epsilon
* **epsilon_decay(str)**: Decay per step for epsilon
#### Options for ValueAgent
If using a DQN you can either have it stocastic or greedy. A greedy DQN chooses a=argmax A(s,a) whereas a stocastic agent uses a boltsmann softmax to give you a=sample Softmax( A(s,a) for all a)
* **deterministic(bool)**: If set to true we have a (normal) greedy DQN. If we set to false we use softmax on the values to get a policy distribution. Note that I am not sure how legit it is to do that and we never used that feature in the paper. Required field!

### Examples of configuration options
This example will randomly select between 3 different agent options at each step. 

#### Example 1
**agents.txt**
```
RandomAgent
PolicyAgent --net=Linear --train=cem --model=linear-cem.pt --save_every=50
ValueAgent --net=Linear --train=dqn --model=linear-dqn --resume=140 --deterministic=True
```
**ensemble.txt**
```
RandomAgent
```

#### Example 2
Example 2 on the other hand will combine a CEM model with random actions that decay based on an epsilon greedy schedule. It will start always choosing one of the random actions and by the end it will be choosing the CEM agent almost every time. 


**Example 2 agents.txt**
```
PolicyAgent --net=Linear --train=cem --model=linear-cem.pt
RandomAgent
```
**Example 2 ensemble.txt**
```
EpsilonGreedy --epsilon_start=1 --epsilon_finish=0.1 --epsilon_decay=0.0001
```
This agent has the full observation space and sees 2 past observations. 4*2=8 input neurons

### Program Output
The program will print some of it's information right into the console (TODO -> Store this in a file as well). This will include most of the debugging print statements as well as things such as "Connection established" messages as well as some training print statements. 

In additon, the program will generate the following data files for analysis
* `data/logs/<time>/logs.csv` -> A csv file generated by the python enviornment interface which stores all data about the episode runs into the CSV file
* `data/logs/<time>/console.txt` -> Prints a copy of the console output to a text file for access later
* `data/logs/<time>/settings.txt` -> A file printing all of the configuration information uses for training including things like the network archtecture and DRL training hyperparaemeters
* `data/logs/<time>/tensorboard` -> A folder to store all of the tensorboard information. To access it navigate to `cd data/logs` and then run the command `tensorboard --host 0.0.0.0 --logdir .` and then open your web browser and go to `http://127.0.0.1:6006` if on same machine or `http://server_ip:6006` to view the logs. 


## Running the python program directly
The program `main.py` connects all the differents parts of the repository into a single program. When executed the program will handle training as well as the enviornment. To run the program simply call `python3.8 main.py` along with some optional command-line arguments. Specified below.

* **debug(bool)**: Specify if data should be printed to the console while running to debug
* **env(string)**: Use to specify which enviornment you will use. Takes an OpenAI flag. 
* **agents_path(string)**: The path to the file specifying which agents to use as well as specifying the training regiment for any DRL algorythems listed. Leave blank for `data/agents.txt`
* **ensemble_path(string)**: The path to a file containing one line which has the configuration for how to choose which agent to use at each step. Some of the otpions for how to select an agent are epsilon-greedy or randomly selecting an agent. Another option is to have another DRL agent choose which agent to use. More information below. Leave blank for `data/ensemble.txt`

## Example runs
To demonstrate that this works I did a few sample runs which are currently in `data/logs` on the enviornment `CartePole-v0`. Here is a picture of the tensorboard training plot for those runs:
![alt text](https://i.ibb.co/LnR6cfz/Screen-Shot-2021-01-05-at-2-05-32-PM.png)
