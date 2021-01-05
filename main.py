#!/usr/bin/env python3
import src.agents as agents
import argparse
import gym
import torch
from src.worker import Worker
import src.worker as worker
import time
from multiprocessing import set_start_method
#from torch.multiprocessing import Process, Queue
from queue import Queue
import traceback
import src.ensemble as ensemble
import src.model as model
import os
import src.train as trainers
import subprocess
import src.plotting as plotting
import sys
from datetime import datetime


"""
    The main function for the program. It takes in parameters and then will train neural networks accordingly
"""
def main():
    #The first part of the main prgoram takes in arguments that are self explaniatory
    print("Program Starting")
    parser = argparse.ArgumentParser()
    #Carte Pole = CartPole-v0

    parser.add_argument('--env', type=str, default='CartPole-v0',help='The ID of the enviornment to use')
    parser.add_argument("--debug", default=False,action='store_true',help='Enable to print debug statements to console')
    parser.add_argument("--ensemble_path", type=str, default='data/ensemble.txt', help="Enter the path/file containing the ensemble configuration")
    parser.add_argument("--agents_path", type=str, default='data/agents.txt', help="Enter the path/file containing the individual agents configuration")
    args = parser.parse_args()
    set_start_method('spawn')
    os.makedirs(logs_folder)
    settings_file = open(logs_folder + '/settings.txt', 'w')
    try:
        settings_file.write('Git Hash: %s\n\n'  % (subprocess.check_output('cat .git/ORIG_HEAD', shell=True).decode()))
    except:
        print("Failed to write Git hash to settings file!")
    settings_file.write('Command Arguments: ' + str(args)+ '\n\n')
    settings_file.write('Agents Configuration File:\n')
    with open(args.agents_path,'r') as agents_file:
        settings_file.writelines(['\t' + x for x in agents_file])
        agents_file.close()
    settings_file.write('\nEnsemble Configuration File: ')
    with open(args.ensemble_path,'r') as ensemble_file:
        settings_file.writelines([x for x in ensemble_file])
        ensemble_file.close()
    settings_file.write('\n\n')

    plotter = plotting.Plotter(logs_folder + "/tensorboard")
    env = gym.make(args.env)
    #The way this works is that it will load all of the agents/ensembles and return their collective data
    env, agent, trainers, batch_queue, episode_queue = set_agents(env, args, settings_file, plotter)
    worker = Worker(agent=agent, debug=True, episode_queue=episode_queue, batch_queue=batch_queue, plotter=plotter)
    settings_file.close()
    #The rest of the code is the infinite loop. It will play an episode in the enviornment and then once an episode is played it will attempt to run all of the trainers that are running
    ex = None
    try:
        episode_num = 0
        while True:
            worker.play_episode(env, episode_num)
            for trainer in trainers:
                trainer.step()
            episode_num+=1
    except Exception as e:
        print("An exception occured! Cleaning up!")
        print(e)
        traceback.print_exc()
        ex = e
    finally:
        print("Closing Enviornment")
        env.close()
        plotter.close()
        settings_file.close()
        if ex is not None:
            raise ex

#Read the agent/ensemble files. First it will read in all of the agents from agents.txt and then it will setup the ensemble to select which agent to use
#In order to train with only one agent (Say just pure CEM) just make sure the agents.txt only has one agent in it and the ensemble.txt is set to just select randomly (out of the only agent available)
def set_agents(env, args, settings_file, plotter):
    agents = []
    trainers = []
    ensemble_episode_queue, ensemble_batch_queue = (ensemble.EnsembleQueue(), ensemble.EnsembleQueue())
    agents_file = open(args.agents_path, 'r')
    for line in agents_file:
        agent, trainer, episode_queue, batch_queue = setup_agent(env, line, args, settings_file, plotter)
        assert agent is not None
        agents.append(agent)
        if trainer is not None:
            trainers.append(trainer)
            ensemble_batch_queue.add_agent_queue(batch_queue)
            ensemble_episode_queue.add_agent_queue(episode_queue)
    env = ensemble.SelectAgentWrapper(env, agents)
    ensemble_file = open(args.ensemble_path, 'r')
    agent, trainer, episode_queue, batch_queue = setup_agent(env, ensemble_file.readline(), args, settings_file, plotter)
    ensemble_file.close()
    if trainer is not None:
        trainers.append(trainer)
        ensemble_batch_queue.set_ensemble_queue(batch_queue)
        ensemble_episode_queue.set_ensemble_queue(episode_queue)
    agents_file.close()
    return env, agent, trainers, ensemble_batch_queue, ensemble_episode_queue

#Read an individual agent from a line. That text ine specifies the configuration for the agent to be loaded
def setup_agent(env, line, args, settings_file, plotter):
    line_parser = argparse.ArgumentParser()
    line_parser.add_argument('agent', type=str, default=None, help='The type of agent. RandomAgent, ValueAgent, PolicyAgent')
    line_parser.add_argument('--net', type=str, help='The type of neural network. Either Linear, LSTM, or ConvNet')
    line_parser.add_argument('--train', type=str, default=None, help='Enter the type of training: cem, a2c, dqn or leave blank to not train')
    line_parser.add_argument('--model', type=str, default=None, help='Enter the name of the model file (located in data/models)')
    line_parser.add_argument('--resume', type=int, default=-1, help='Enter this flag in order to resume training of a previous model. Enter the iter you are resuming from.')
    line_parser.add_argument('--save_every', type=int, default=None, help='Enter this flag to specify how many iterations before we create a backup of the model. Leave blank for no backups of the model.')
    line_parser.add_argument("--epsilon", type=float, default=0, help='Use epsilon in a Value or Policy agent to ensure sufficient exploration')
    line_parser.add_argument("--epsilon_start", type=float, default=None, help='The starting value for epsilon (ONLY USE IF agent=EpsilonGreedy)')
    line_parser.add_argument("--epsilon_finish", type=float, default=None, help='The final value of epsilon (ONLY USE IF agent=EpsilonGreedy)')
    line_parser.add_argument("--epsilon_decay", type=float, default=None, help='The decay value for epsilon (per step) (ONLY USE IF agent=EpsilonGreedy)')
    line_parser.add_argument("--fixed_action", type=int, default=None, help='The fixed action for FixedAgent')
    line_parser.add_argument("--deterministic", type=bool, default=None, help='If using a DQN set this flag to make it deterministic, otherwise it will use softmax on the Q values to get a policy')
    line_parser.add_argument("--net_config",type=str,default=None,help="If Linear: Specify the hidden layer, if LSTM specify as:hidden_units,num_layers")
    config = line_parser.parse_known_args(line.replace('\n', '').split(' '))[0]
    assert config.agent is not None
    agent = None
    trainer = None
    episode_queue = batch_queue = None
    tag = line.replace('-', '').replace('=', '_').replace('\n', '').replace('.', '_')
    if config.agent == 'RandomAgent':
        agent = agents.RandomAgent(env.action_space.n)

    if config.agent == 'FixedAgent':
        assert config.fixed_action is not None
        agent = agents.FixedAgent(env, config.fixed_action)
    else:
        assert config.fixed_action is None

    if config.agent == 'EpsilonGreedy':
        agent = agents.EpsilonGreedy(env, agents.Epsilon(config.epsilon_start, config.epsilon_finish, config.epsilon_decay))
    else:
        assert config.epsilon_start is None and config.epsilon_finish is None and config.epsilon_decay is None

    assert config.agent == 'ValueAgent' or config.deterministic == None#Deterministic flag can only be used with DQN
    if config.agent in ['ValueAgent', 'PolicyAgent']:
        if config.net_config is None:
            net = getattr(model, config.net)(env.observation_space.shape, env.action_space.n)
        else:
            if config.net == 'LSTM':
                net_config = [int(s) for s in config.net_config.replace("\"", '').split(',')]
                net = model.LSTM(env.observation_space.shape, env.action_space.n, hidden_units=net_config[0], num_layers=net_config[1])
            elif config.net == 'Linear':
                net = model.Linear(env.observation_space.shape, env.action_space.n, hidden_layer=int(config.net_config))
            else:
                assert False#Cannot use net config for any other network type!
        #Ensuring that the model exists if resuming or that the model does not exist already if starting from scratch
        model_path = 'data/models/' + config.model
        assert not (config.resume >= 0 and not os.path.exists(model_path))
        assert not (config.resume == -1 and os.path.exists(model_path))
        #Ensure that either we have a model to use or that we are training a model. We cannot test a model that does not exist!
        assert config.resume >= 0 or config.train is not None

        #Setting the proper model path and loading it if required
        net.set_path(model_path=model_path)
        if config.resume >= 0:net.load(model_path=model_path)
        write_folder = logs_folder + '/agent_logs'
        if not os.path.exists(write_folder):os.makedirs(write_folder)
        write_file = write_folder + '/' + tag + '.csv'
        if config.agent == 'PolicyAgent':
            agent = agents.PolicyAgent(net, tag, write_file,epsilon=config.epsilon)
        else:
            assert config.deterministic is not None
            agent = agents.ValueAgent(net, tag, write_file, deterministic=config.deterministic, epsilon=config.epsilon)

        assert config.model is not None
        if config.train is not None:
            assert config.train in ['cem', 'a2c', 'dqn']
            backup_file = logs_folder + '/models/' + config.model
            if not os.path.exists(logs_folder + '/models'):os.makedirs(logs_folder + '/models')

            episode_queue, batch_queue = Queue(), Queue()
            trainer = {'cem':trainers.CEMTrainer, 'dqn':trainers.DQNTrainer, 'a2c':trainers.PolicyTrainer}[config.train] \
                (agent=agent, episode_queue=episode_queue, batch_queue=batch_queue, plotter=plotter, debug=True, iter_num=config.resume if config.resume > 0 else 0, backup_file=backup_file)
            settings_file.write("Trainer=" + str(trainer.__dict__) + "\n")
    else:
        assert config.train is None and config.model is None and config.net is None and config.save_every is None and config.epsilon < 0.0001 and config.net_config is None
    assert agent is not None#If it was an invalid option, check if there was a blank line in the agents.txt or ensemble.txt files
    agent.tag = tag
    #If we want to print tensorboard but there is no trainer then we create a DummyTrainer that just prints but does nothing else
    settings_file.write('Agent (' + line + '): Agent=' + str(agent.__dict__) + "\n\n")
    return agent, trainer, episode_queue, batch_queue


def cleanup():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    logs_folder = 'data/logs/' + datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("error")
    main()
    cleanup()