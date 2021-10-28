"""
# **************************<><><><><>********************************
# * Script for the Simple Perception-Action Firefighting Environment *
# **************************<><><><><>********************************
#
# This script and all its dependencies are implemented by: Esmaeil Seraj
#   - Esmaeil Seraj, PhD Student, Institute for Robotics and Inteelligent
#   Machines (IRIM), Electrical and Computer Engineering (ECE)
#   Georgia Tech, Atlanta, GA, USA
#   - email <eseraj3@gatech.edu>
#
# Supported by Python 3.6.4
#
"""

import numpy as np
import matplotlib.pyplot as plt
from FireCommander_Base_Utilities import EnvUtilities
from WildFire_Model import WildFire
import time

Agent_Util = EnvUtilities()


# Simplified FireCommander Environment
class FireCommanderEasy(object):
    def __init__(self, world_size=None, duration=None, fireSpots_Num=None, P_agent_num=None, A_agent_num=None, vision=None,
                 rnd_seed=None, center_init=None, stationary_fire=None, local_reward_ratio=None, termination_rewrad=None):

        # pars parameters
        self.center_init = False if center_init is None else center_init  # flag if you want to initialize agents in center (rnd init if NOT)
        self.stationary_fire = False if stationary_fire is None else stationary_fire  # stationary fire loci during training
        self.rnd_seed = 1 if rnd_seed is None else rnd_seed  # initialize random seed
        self.vision = 1 if vision is None else vision  # visible hops around each agent
        self.world_size = 10 if world_size is None else world_size  # world size
        self.duration = 100 if duration is None else duration  # numbr of steps per game
        self.local_reward_ratio = 0.5 if local_reward_ratio is None else local_reward_ratio  # the local reward ration
        self.termination_rewrad = False if termination_rewrad is None else termination_rewrad  # termination reward flag
        self.fireSpots_Num = 5 if fireSpots_Num is None else fireSpots_Num  # number of firespots
        self.perception_agent_num = 2 if P_agent_num is None else P_agent_num  # number of perception agents
        self.action_agent_num = 2 if A_agent_num is None else A_agent_num  # number of action agents

    # initialize the environment
    def env_init(self, comm_range=None, water_dump_action=None, no_op_action=None, feat_dim=None):
        # initialize random seed
        # np.random.seed(self.rnd_seed)

        # initialize some required variables
        self.comm_hop = 5 if comm_range is None else comm_range  # number of hops for discrete communication range:: default=5
        self.water_dump_action = True if water_dump_action is None else water_dump_action  # flag if the water dumping is an action or not
        self.no_op_action = [False, False] if no_op_action is None else no_op_action  # flag if no-op is going to be an action
        self.done = False  # initialize the END flag
        self.outcom_flg = False  # initialize the outcom seccess flag
        self.FOV_list = []  # FOV buffer
        self.sensed_list = np.array([])  # the sensed list, store the points currently on fire and have been sensed by agents
        self.pruned_list = np.array([])  # the pruned list, store the pruned fire spots
        self.sensed_contribution = [0] * self.perception_agent_num  # keeping track of P agents' contributions
        self.pruned_contribution = [0] * self.action_agent_num  # keeping track of A agents' contributions
        self.feat_dim = 4 + (2 * self.perception_agent_num) + self.action_agent_num if feat_dim is None else feat_dim

        # initialize the state matrix
        # state matrix - contents are as follow:
        # Dim (0):: fire_map [0 -> not_on_fire, 1 -> onFire_notFound, 2 -> onFire_found, 3 -> pruned]
        # Dim (1) to (num_PAgents):: individual P_agent maps [0 -> nothing, 1 -> agent's location, 2 -> agent's scope]
        # Dim (num_PAgents) to (num_PAgents+num_AAgents):: individual A_agent maps [0 -> nothing, 1 -> agent's location]
        self.state = np.zeros([self.perception_agent_num + self.action_agent_num + 1, self.world_size, self.world_size], dtype=float)

        # initialize firespot positions (uniformly randomly distributed)
        if not self.stationary_fire:
            self.firespot_loci = np.random.choice(np.arange(0, self.world_size * self.world_size), size=self.fireSpots_Num, replace=False)
        else:
            self.firespot_loci = np.array([39, 5, 22, 81, 69])  # some random fixed position within bounds for testing

        # initialize agents (initialized in the middle of terrain)
        self.agent_pose_dim = 2  # 2D coordinates
        self.agent_state = []  # agents state matrix format:: [x, y, type] -> Perception.type=0 & Action.type=1
        
        if self.center_init:
            for i in range(self.perception_agent_num):
                self.agent_state.append([int(self.world_size / 2), int(self.world_size / 2), 0])  # Perception Agent
            for i in range(self.action_agent_num):
                self.agent_state.append([int(self.world_size / 2), int(self.world_size / 2), 1])  # Action Agent
        else:
            temp = np.random.choice(np.arange(0, self.world_size),
                                    size=(2, self.perception_agent_num + self.action_agent_num),
                                    replace=False)  # randomize agents initial positions

            for i in range(self.perception_agent_num):
                self.agent_state.append([temp[0][i], temp[1][i], 0])  # Perception Agent
            for i in range(self.action_agent_num):
                self.agent_state.append([temp[0][i + self.perception_agent_num], temp[1][i + self.perception_agent_num], 1])  # Action Agent

        # update state matrix
        self.state_matrix_update()

        # initialize the adjacency matrices
        self.adjacent_agents_PnP, self.adjacent_agents_PnA, self.adjacent_agents_AnA = self.get_adjacency_matrices()

        # initialize task success rates
        self.perception_complete, self.action_complete = 0.0, 0.0

        # initialize rewards
        self.old_reward_without_adjacent = 0.0
        self.global_reward = 0.0
        self.reward = 0.0

    # proceed the environment one step into the future
    def env_step(self, action, step=0, a_c_threshold=1.0, r_func=None, global_penalty=-0.1, local_P_reward=0.1, local_A_reward=0.1,
                 A_penalty=-0.05, max_steps=1000):
        # update the firespot positions (in current simple FireCommander firespots are left stationary, thus commented)
		# to include moving firespoots (fire propoagation), consider allowing firepropagation using the line below (line 113)
		# make sure to check the format of the propagate_fire() function from the Wildfire_Model class and modify accordingly
        # self.firespot_loci = propagate_fire()

        # update agents' states
        self.agent_state_update(action)

        # update the state space
        self.state_matrix_update()

        # update the Perception agents' contribution
        self.FOV_list = []
        for i in range(self.perception_agent_num):
            sensed_num_prev = self.sensed_contribution[i]  # remember how many spots have been detected so far
            this_sensed_list, FOV = Agent_Util.sensing(self.state[0].copy(), self.state[i + 1], self.world_size, self.vision, self.state.copy(),
                                                       self.feat_dim, self.perception_agent_num, self.action_agent_num)  # sensing
            self.sensed_list = np.unique(np.concatenate((self.sensed_list, this_sensed_list), axis=0))  # adding new sensing results to the list
            self.FOV_list.append(FOV)  # gather FOVs in a list for CNN input
            self.sensed_contribution[i] += len(list(this_sensed_list)) - sensed_num_prev  # compute contributions

        # update the Action agents' contribution
        for i in range(self.perception_agent_num, self.perception_agent_num + self.action_agent_num):
            if (action[i] == 4) and self.water_dump_action:  # perform pruning only if agents action says so
                pruned_num_prev = self.pruned_contribution[i - self.perception_agent_num]  # remember how many spots have been pruned so far
                this_pruned_list, self.sensed_list, self.firespot_loci = Agent_Util.pruning(self.state[0].copy(), self.state[i + 1], self.sensed_list,
                                                                                            self.firespot_loci, self.world_size)  # pruning
                self.pruned_list = np.concatenate((self.pruned_list, this_pruned_list), axis=0)  # adding new sensing results to the list
                self.pruned_contribution[i - self.perception_agent_num] += len(list(this_pruned_list)) - pruned_num_prev  # compute contributions
            elif not self.water_dump_action:
                pruned_num_prev = self.pruned_contribution[i - self.perception_agent_num]  # remember how many spots have been pruned so far
                this_pruned_list, self.sensed_list, self.firespot_loci = Agent_Util.pruning(self.state[0].copy(), self.state[i + 1], self.sensed_list,
                                                                                            self.firespot_loci, self.world_size)  # pruning
                self.pruned_list = np.concatenate((self.pruned_list, this_pruned_list), axis=0)  # adding new sensing results to the list
                self.pruned_contribution[i - self.perception_agent_num] += len(list(this_pruned_list)) - pruned_num_prev  # compute contributions
            else:
                continue

        # update the state space
        self.state_matrix_update()

        # update the adjacency matrices
        self.adjacent_agents_PnP, self.adjacent_agents_PnA, self.adjacent_agents_AnA = self.get_adjacency_matrices()

        # compute rewards
        all_adjacencies = self.adjacent_agents_PnP + self.adjacent_agents_AnA + self.adjacent_agents_PnA

        # compute the global reward
        if r_func == 'RF1':
            self.global_reward = self.get_global_reward1(sum(self.sensed_contribution), sum(self.pruned_contribution), all_adjacencies)
        elif r_func == 'RF2':
            self.global_reward = self.get_global_reward2(len(list(self.firespot_loci)))
        elif r_func == 'RF3':
            self.global_reward = self.get_global_reward3(len(list(self.firespot_loci)), action, global_penalty=global_penalty, A_penalty=A_penalty)
        elif r_func is None:
            self.global_reward = self.get_global_reward3(len(list(self.firespot_loci)), action, global_penalty=global_penalty, A_penalty=A_penalty)
        else:
            raise ValueError(">>> Oops! The specified Global Reward Function doesn't exist. Options: RF1, RF2, RF3")

        # compute the local reward
        self.local_reward = self.get_local_reward(
            self.sensed_contribution, self.pruned_contribution, len(list(self.firespot_loci)), action, local_P_reward=local_P_reward,
            local_A_reward=local_A_reward, A_penalty=A_penalty, global_penalty=global_penalty)

        # calculate the mixed reward with local reward ratio
        local_ratio_list = [self.local_reward_ratio] * (self.perception_agent_num + self.action_agent_num)
        global_ratio_list = [1 - self.local_reward_ratio] * (self.perception_agent_num + self.action_agent_num)
        global_reward_list = [self.global_reward] * (self.perception_agent_num + self.action_agent_num)

        global_reward_list_scaled = [x * y for x, y in zip(global_ratio_list, global_reward_list)]
        local_reward_list_scaled = [x * y for x, y in zip(local_ratio_list, self.local_reward)]

        self.reward = [x + y for x, y in zip(local_reward_list_scaled, global_reward_list_scaled)]

        # compute performances
        # perception performance:: (sensed + pruned) / (active + pruned)
        self.perception_complete = \
            (len(list(self.sensed_list)) + len(list(self.pruned_list))) / (len(list(self.firespot_loci)) + len(list(self.pruned_list)))

        # action performance::  (pruned) / (active + pruned)
        self.action_complete = len(list(self.pruned_list)) / (len(list(self.firespot_loci)) + len(list(self.pruned_list)))
        # if all the firespots have been put out, finish the game
        if self.action_complete >= a_c_threshold:
            self.done = True
            self.outcom_flg = True
            if self.termination_rewrad:
                episode_return = [10.0 * self.fireSpots_Num] * (self.perception_agent_num + self.action_agent_num)
                self.reward = [x + y for x, y in zip(self.reward, episode_return)]

        # step forward in game
        step += 1

        # if a certain number of steps are passed but no good results are achieved, break since the training is not going well
        if not self.done and (step >= max_steps):
            self.done = True
            if self.termination_rewrad:
                episode_return = [-10.0 * (self.fireSpots_Num - len(list(self.pruned_list)))] * (self.perception_agent_num + self.action_agent_num)
                self.reward = [x + y for x, y in zip(self.reward, episode_return)]

        return self.state, self.reward, self.done, self.outcom_flg, step, self.perception_complete, self.action_complete

    # generating the state matrix
    def state_matrix_update(self):
        # clean-up the previous states
        self.state = np.zeros([self.perception_agent_num + self.action_agent_num + 1, self.world_size, self.world_size], dtype=float)

        # update fire's state in state matrix (Dim 1)
        self.state[0].reshape(1, self.world_size * self.world_size)[0, self.firespot_loci.astype(int)] = 1
        self.state[0].reshape(1, self.world_size * self.world_size)[0, self.sensed_list.astype(int)] = 2
        self.state[0].reshape(1, self.world_size * self.world_size)[0, self.pruned_list.astype(int)] = 3

        # update agents states in state matrix
        # P agents
        if self.vision == 1:
            for i in range(1, self.perception_agent_num + 1):
                # mark P agents' scopes (1-hop)
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), self.agent_state[i - 1][1]] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), self.agent_state[i - 1][1]] = 2
                self.state[i, self.agent_state[i - 1][0], min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2
                self.state[i, self.agent_state[i - 1][0], max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2

                # P agent location
                self.state[i, self.agent_state[i - 1][0], self.agent_state[i - 1][1]] = 1
        elif self.vision == 2:
            for i in range(1, self.perception_agent_num + 1):
                # mark P agents' scopes (2-hop)
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), self.agent_state[i - 1][1]] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), self.agent_state[i - 1][1]] = 2
                self.state[i, self.agent_state[i - 1][0], min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2
                self.state[i, self.agent_state[i - 1][0], max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2

                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 2), self.agent_state[i - 1][1]] = 2
                self.state[i, self.agent_state[i - 1][0], min(self.world_size - 1, self.agent_state[i - 1][1] + 2)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 2), self.agent_state[i - 1][1]] = 2
                self.state[i, self.agent_state[i - 1][0], max(0, self.agent_state[i - 1][1] - 2)] = 2

                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 2), min(self.world_size - 1, self.agent_state[i - 1][1] + 2)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 2), max(0, self.agent_state[i - 1][1] - 2)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 2), max(0, self.agent_state[i - 1][1] - 2)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 2), min(self.world_size - 1, self.agent_state[i - 1][1] + 2)] = 2

                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 2), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 2), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 2), max(0, self.agent_state[i - 1][1] - 1)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 2), min(self.world_size - 1, self.agent_state[i - 1][1] + 1)] = 2

                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 2)] = 2
                self.state[i, min(self.world_size - 1, self.agent_state[i - 1][0] + 1), max(0, self.agent_state[i - 1][1] - 2)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), max(0, self.agent_state[i - 1][1] - 2)] = 2
                self.state[i, max(0, self.agent_state[i - 1][0] - 1), min(self.world_size - 1, self.agent_state[i - 1][1] + 2)] = 2

                # P agent location
                self.state[i, self.agent_state[i - 1][0], self.agent_state[i - 1][1]] = 1
        else:
            raise ValueError(">>> Oops! Sorry, for now, only 1-hop and 2-hop visions are available options...")

        # A agents
        for i in range(self.perception_agent_num + 1, self.perception_agent_num + self.action_agent_num + 1):
            self.state[i, self.agent_state[i - 1][0], self.agent_state[i - 1][1]] = 1  # A agents locations

    # updating agents' states according to the actions taken
    def agent_state_update(self, action_type):
        # Update P agents' states
        for i in range(self.perception_agent_num):
            if action_type[i] == 0:  # Action: Forward
                self.agent_state[i][0] = max(self.agent_state[i][0] - 1, 0)
            elif action_type[i] == 1:  # Action: Backward
                self.agent_state[i][0] = min(self.agent_state[i][0] + 1, self.world_size - 1)
            elif action_type[i] == 2:  # Action: Left
                self.agent_state[i][1] = max(self.agent_state[i][1] - 1, 0)
            elif action_type[i] == 3:  # Action: Right
                self.agent_state[i][1] = min(self.agent_state[i][1] + 1, self.world_size - 1)
            elif (action_type[i] == 4) and self.no_op_action[0]:  # Action: No-op
                continue

        # Update A agents' states
        for i in range(self.perception_agent_num, self.perception_agent_num + self.action_agent_num):
            if action_type[i] == 0:  # Action: Forward
                self.agent_state[i][0] = max(self.agent_state[i][0] - 1, 0)
            elif action_type[i] == 1:  # Action: Backward
                self.agent_state[i][0] = min(self.agent_state[i][0] + 1, self.world_size - 1)
            elif action_type[i] == 2:  # Action: Left
                self.agent_state[i][1] = max(self.agent_state[i][1] - 1, 0)
            elif action_type[i] == 3:  # Action: Right
                self.agent_state[i][1] = min(self.agent_state[i][1] + 1, self.world_size - 1)
            elif action_type[i] == 4:  # Action: Dump Extinguisher
                continue
            elif (action_type[i] == 5) and self.no_op_action[1]:  # Action: No-op
                continue

    # get the reward for agents (reward function number 1) - w/o time penalty, w/o firespot penalty, w/ communication reward
    def get_global_reward1(self, num_sensed, num_pruned, all_adjacencies):
        # compute performance rewards
        new_reward = 2.0 * num_sensed
        new_reward += 20.0 * num_pruned

        im_reward = new_reward - self.old_reward_without_adjacent
        self.old_reward_without_adjacent = new_reward

        # computing adjacency rewards for homogeneous communicatiion channel
        uniques = list(range(0, len(self.agent_state)))
        for pair in all_adjacencies:
            for idx in pair:
                if idx in uniques:
                    uniques.remove(idx)
                    if len(uniques) == 0:
                        break

        im_reward -= 1.0 * len(uniques)

        # computing adjacency rewards for homogeneous communicatiion channel
        uniques_hetero_cm = list(range(0, len(self.agent_state)))
        for pair in all_adjacencies:
            if 0 <= pair[0] < self.perception_agent_num:  # one way check is OK since heterogeneous communication is symmetric
                if self.perception_agent_num <= pair[1] < self.perception_agent_num + self.action_agent_num:
                    if pair[0] in uniques_hetero_cm:
                        uniques_hetero_cm.remove(pair[0])
                    if pair[1] in uniques_hetero_cm:
                        uniques_hetero_cm.remove(pair[1])
                    if len(uniques_hetero_cm) == 0:
                        break

        im_reward -= 3.0 * len(uniques_hetero_cm)

        return im_reward

    # get the reward for agents (reward function number 2) - w/ time penalty, w/o firespots penalty, w/o communication reward
    @staticmethod
    def get_global_reward2(num_firespots):
        # computing performance rewards
        if num_firespots > 0:
            im_reward = -1.0
        else:
            im_reward = 0.0

        return im_reward

    # get the reward for agents (reward function number 2) - w/o time penalty, w/ firespots penalty, w/o communication reward
    def get_global_reward3(self, num_firespots, action, global_penalty=-0.1, A_penalty=-0.05):
        # computing performance rewards
        im_reward = global_penalty * num_firespots

        # penalty for water dumping action
        for i in range(self.perception_agent_num, self.perception_agent_num + self.action_agent_num):
            if action[i] == 4:
                im_reward += A_penalty

        return im_reward

    # get the reward for agents (reward function number 1) - w/o time penalty, w/o firespot penalty, w/ communication reward
    def get_local_reward(self, sensing_contributions, action_contributions, num_firespots, action, local_P_reward=0.1, local_A_reward=0.1,
                         A_penalty=-0.05, global_penalty=-0.1):
        # compute performance rewards
        if num_firespots > 0:
            im_reward_list = [global_penalty] * len(action)
        else:
            im_reward_list = [0.0] * len(action)
        self.Perception_reward = [x * y for x, y in zip([local_P_reward] * len(sensing_contributions), sensing_contributions)]
        self.Action_reward = [x * y for x, y in zip([local_A_reward] * len(action_contributions), action_contributions)]

        # penalty for water dumping action
        A_actions = action[len(sensing_contributions):]
        for i in range(len(action_contributions)):
            if A_actions[i] == 4:
                self.Action_reward[i] += A_penalty

        local_temp = self.Perception_reward + self.Action_reward
        self.local_reward = [x + y for x, y in zip(local_temp, im_reward_list)]

        return self.local_reward

    # generate the adjacency matrix for current time-step
    def get_adjacency_matrices(self):
        self.adjacent_agents_PnP, self.adjacent_agents_PnA, self.adjacent_agents_AnA = [], [], []  # clean-up the previous adjacencies
        # Perception-Perception adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.perception_agent_num):
                if i != j:
                    pose1 = self.agent_state[i]
                    pose2 = self.agent_state[j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim], pose2[0:self.agent_pose_dim], hop_num=self.comm_hop):
                        self.adjacent_agents_PnP.append([i, j])
        # Perception-Action adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.action_agent_num):
                pose1 = self.agent_state[i]
                pose2 = self.agent_state[self.perception_agent_num + j]
                if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim], pose2[0:self.agent_pose_dim], hop_num=self.comm_hop):
                    self.adjacent_agents_PnA.append([i, self.perception_agent_num + j])
        # Action-Action adjacency
        for i in range(self.action_agent_num):
            for j in range(self.action_agent_num):
                if i != j:
                    pose1 = self.agent_state[self.perception_agent_num + i]
                    pose2 = self.agent_state[self.perception_agent_num + j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim], pose2[0:self.agent_pose_dim], hop_num=self.comm_hop):
                        self.adjacent_agents_AnA.append([self.perception_agent_num + i, self.perception_agent_num + j])

        return self.adjacent_agents_PnP, self.adjacent_agents_PnA, self.adjacent_agents_AnA

    # get the FOV of each agent
    def get_FOV(self):
        if len(self.FOV_list) > 0:
            return self.FOV_list
        else:
            FOV_list = []

            for i in range(self.perception_agent_num):
                # extracting 1-hot encoded FOV tensors
                FOV = Agent_Util.FOV_encoding(self.state.copy(), self.state[i + 1], self.feat_dim, self.perception_agent_num,
                                              self.action_agent_num, self.world_size, self.vision)
                # creating the FOV list
                FOV_list.append(FOV)

            return FOV_list

    '''Get the FOV of each agent in vector form.

    For each square information is:
        [[NxN position one hot], [F+P+A binary encoding of agents on square], [out of bounds], [seen before]]
    '''
    def get_FOV_vectorized(self):
        FOV_list = []

        for i in range(self.perception_agent_num):
            # extracting 1-hot encoded FOV tensors
            FOV = Agent_Util.FOV_vectorized_encoding(self.state.copy(), 
                self.state[i + 1], self.feat_dim, self.perception_agent_num,
                self.action_agent_num, self.world_size, self.vision)

            # creating the FOV list
            FOV_list.append(FOV)

        return FOV_list

    def get_agent_state_vector(self):
        agent_state_vector = []

        for state in self.agent_state:
            agent_state_vector.append([Agent_Util.position_one_hot(state[0], state[1], self.world_size).tolist(), state[2]])

        return agent_state_vector

    # close env
    @staticmethod
    def env_close():
        return 0


if __name__ == '__main__':
    # initialize the env
    env = FireCommanderEasy(vision=1, termination_rewrad=False, local_reward_ratio=1.0)
    water_dump_action = False
    no_op_action = [False, False]
    env.env_init(water_dump_action=water_dump_action, no_op_action=no_op_action)
    visualize = False

    # parameters
    Num_P = 2
    Num_A = 2
    P_action_space = 4
    A_action_space = 4
    if water_dump_action:
        A_action_space += 1
    if no_op_action[0]:
        P_action_space += 1
    if no_op_action[1]:
        A_action_space += 1

    # go through episodes of the game
    startTime = time.time()
    num_episode = 2000
    for t in range(num_episode):
        action_p = np.random.randint(0, P_action_space, Num_P)  # Perception agent action generator, using randint now
        action_a = np.random.randint(0, A_action_space, Num_A)  # Action agent action generator, using randint now

        actions = list(action_p) + list(action_a)
        print(actions)

        state, reward, done, outcom_flg, t, p_c, a_c = env.env_step(actions, step=t, a_c_threshold=1.0, r_func='RF3', global_penalty=-0.1,
                                                                    local_P_reward=0.1, local_A_reward=0.1, A_penalty=-0.05, max_steps=1000)
        FOVs_list = env.get_FOV()

        print('step:: ' + str(t) + '     reward:: ' + str(reward) + '     p_c:: ' + str(p_c) + '     a_c:: ' + str(a_c))

        # end the game if all firespots are "put out"
        if done and outcom_flg:
            print('success')
            break
        if done and not outcom_flg:
            print('failure')
            break

        # visualization
        if visualize:
            combined_states = np.zeros((10, 10))
            for ii in range(state.shape[0]):
                combined_states += state[ii]
            plt.imshow(combined_states)
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    env.env_close()
