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
# Supported by Python 3.6.4 and PyGame 1.9.6 (or any later version)
#
"""

import pygame  # for online visualization
from pygame.locals import *
import numpy as np
import random
import matplotlib.pyplot as plt
from WildFire_Model import WildFire
from FireCommander_Cmplx2_Utilities import EnvUtilities

Agent_Util = EnvUtilities()


# Full FireCommander Environment with Battery and Tanker Capacity Limitations
class FireCommanderExtreme(object):
    def __init__(self, world_size=None, duration=None, fireAreas_Num=None, P_agent_num=None, A_agent_num=None, online_vis=False):

        # pars parameters
        self.world_size = 100 if world_size is None else world_size            # world size
        self.duration = 200 if duration is None else duration                  # numbr of steps per game
        self.fireAreas_Num = 2 if fireAreas_Num is None else fireAreas_Num     # number of fire areas
        self.perception_agent_num = 2 if P_agent_num is None else P_agent_num  # number of perception agents
        self.action_agent_num = 2 if A_agent_num is None else A_agent_num      # number of action agents

        # fire model parameters
        areas_x = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
        areas_y = np.random.randint(20, self.world_size - 20, self.fireAreas_Num)
        area_delays = [0] * self.fireAreas_Num
        area_fuel_coeffs = [5] * self.fireAreas_Num
        area_wind_speed = [5] * self.fireAreas_Num
        area_wind_directions = []
        area_centers = []
        num_firespots = []

        for i in range(self.fireAreas_Num):
            area_centers.append([areas_x[i], areas_y[i]])
            num_firespots.append(np.random.randint(low=5, high=15, size=1)[0])
            area_wind_directions.append(random.choice([0, 45, 90, 135, 180]))
        self.fire_info = [area_centers,            # [[area1_center_x, area1_center_y], [area2_center_x, area2_center_y], ...],
                          [num_firespots,          # [[num_firespots1, num_firespots2, ...],
                           area_delays,            # [area1_start_delay, area2_start_delay, ...],
                           area_fuel_coeffs,       # [area1_fuel_coefficient, area2_coefficient, ...],
                           area_wind_speed,        # [area1_wind_speed, area2_wind_speed, ...],
                           area_wind_directions,   # [area1_wind_direction, area2_wind_direction, ...],
                           1.25,                   # temporal penalty coefficient,
                           0.1,                    # fire propagation weight,
                           90,                     # Action Pruning Confidence Level (In percentage),
                           80,                     # Hybrid Pruning Confidence Level (In percentage),
                           1]]                     # mode]

        # the number of stacked frames for training
        self.stack_num = 4

        # initialize the pygame for online visualization
        if online_vis:
            pygame.init()
            # The simulation time is counted in seconds, while the actual time is counted in milliseconds
            clock = pygame.time.Clock()
            # Create a screen (Width * Height) = (1024 * 1024)
            self.screen = pygame.display.set_mode((self.world_size, self.world_size), 0, 32)

    # initialize the environment
    def env_init(self, comm_range=30, init_alt=10):
        # state matrix: 0 -> not_on_fire, 1 -> sensed_on_fire, 2 -> pruned, 3 -> P_agent_loc, 4 -> P_agent_scope, 5 -> A_agent_loc, 6 -> A_agent_scope
        self.state = np.zeros((self.world_size, self.world_size), dtype=float)  # initialize the full state matrix

        # initialize the agents' (e.g., robots') states. Format:: [X, Y, Z, type], where type=0 is Perception and type=1 is Action
        # agaents initialized at the top-right corner (-/), for top-left (//), for bottom-right (--), for bottom-left (/-)
        # self.agent_state = [[int(self.world_size-10), int(self.world_size/10), 10, 0],  # Perception Agent #1
        #                     [int(self.world_size-10), int(self.world_size/10), 10, 0],  # Perception Agent #2
        #                     [int(self.world_size-10), int(self.world_size/10), 10, 1],  # Action Agent #1
        #                     [int(self.world_size-10), int(self.world_size/10), 10, 1]]  # Action Agent #2
        self.agent_state = []
        for i in range(self.perception_agent_num):
            self.agent_state.append([int(self.world_size-10), int(self.world_size/10), init_alt, 0])  # Perception Agent
        
        for i in range(self.action_agent_num):
            self.agent_state.append([int(self.world_size-10), int(self.world_size/10), init_alt, 1])  # Action Agent
        
        self.comm_hop = comm_range  # number of hops for discrete communication range
        self.agent_pose_dim = 3  # 3D coordinates

        # initialize the episode transaction
        self.episode = 0

        # initialize the list to store the target info
        self.target_onFire_list = []
        self.target_info = []

        # initialize the fire region info
        self.fire_init()

        # the END flag
        self.done = False

        # keeping track of neighboring agents (works with both 2D and 3D positions)
        self.adjacent_agents_PnP = []
        self.adjacent_agents_PnA = []
        self.adjacent_agents_AnA = []
        # Perception-Perception adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.perception_agent_num):
                if i != j:
                    pose1 = self.agent_state[i]
                    pose2 = self.agent_state[j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                        self.adjacent_agents_PnP.append([i, j])
        # Perception-Action adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.action_agent_num):
                pose1 = self.agent_state[i]
                pose2 = self.agent_state[self.perception_agent_num + j]
                if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                    self.adjacent_agents_PnA.append([i, self.perception_agent_num + j])
        # Action-Action adjacency
        for i in range(self.action_agent_num):
            for j in range(self.action_agent_num):
                if i != j:
                    pose1 = self.agent_state[self.perception_agent_num + i]
                    pose2 = self.agent_state[self.perception_agent_num + j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                        self.adjacent_agents_AnA.append([self.perception_agent_num + i, self.perception_agent_num + j])

        # task complete info
        self.perception_complete = 0
        self.action_complete = 0

        # initialize rewards
        self.reward = 0.0
        self.Perception_reward = 0
        self.Action_reward = 0

        # initialize the frame that stores the most recent 4 frames for training
        self.frame = []

        # video_replay reconstruction Buffer intitialization
        self.sensed_List_Buffer = []
        self.pruned_List_Buffer = []
        self.agent_state_buffer = []
        self.reward_buffer = []
        self.old_reward_without_adjacent = 0.0

        # FOV buffer
        self.FOV_list = []

    # proceed the environment one step into the future
    def env_step(self, action, p_vel=5, a_vel=5, vert_vel=2, min_alt=5, max_alt=15, time_passed=1, a_c_threshold=1.5, r_func=None):
        self.FOV_list = []
        
        self.agent_state_update(action, p_vel=p_vel, a_vel=a_vel, vert_vel=vert_vel, min_alt=min_alt, max_alt=max_alt)  # update the agents' states
        self.fire_propagation()          # propagate the fire

        # updating the Perception agents' contribution
        for i in range(self.perception_agent_num):
            sensed_num_prev = len(self.sensed_List)
            # Sensing
            self.sensed_List, FOV = Agent_Util.fire_Sensing(self.onFire_List, self.agent_state[i], self.sensed_List, self.world_size)
            self.FOV_list.append(FOV)
            # compute the per-agent contribution (variation of the sensing list size)
            self.sensed_contribution[i] += len(self.sensed_List) - sensed_num_prev

        # updating the Action agents' contribution
        for i in range(self.perception_agent_num, self.perception_agent_num + self.action_agent_num):
            if action[i] == 4:
                pruned_num_prev = len(self.pruned_List)
                # Pruning
                self.onFire_List, self.sensed_List, self.pruned_List, self.new_fire_front, self.target_onFire_list =\
                    Agent_Util.fire_Pruning(self.agent_state[i], self.onFire_List, self.sensed_List, self.pruned_List, self.new_fire_front,
                                            self.target_onFire_list, self.target_info, self.world_size, 0.8)
                # compute the per-agent contribution (variation of the pruned list size)
                self.pruned_contribution[i - self.perception_agent_num] += len(self.pruned_List) - pruned_num_prev
            else:
                continue

        # updating the full state matrix
        state = self.state_gen()
        '''
        # update the frame stack in the FIFO policy - not used currently
        if self.stack_num <= len(self.frame):
            self.frame = self.frame[1:] + [state]
        else:
            self.frame.append(state)
        '''

        # TODO: DISCRETE ADAJACENCY CHECK ############################################################################################################
        # determining the neighboring agents (works with both 2D and 3D positions)
        self.adjacent_agents_PnP = []
        self.adjacent_agents_PnA = []
        self.adjacent_agents_AnA = []
        # Perception-Perception adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.perception_agent_num):
                if i != j:
                    pose1 = self.agent_state[i]
                    pose2 = self.agent_state[j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                        self.adjacent_agents_PnP.append([i, j])
        # Perception-Action adjacency
        for i in range(self.perception_agent_num):
            for j in range(self.action_agent_num):
                pose1 = self.agent_state[i]
                pose2 = self.agent_state[self.perception_agent_num + j]
                if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                    self.adjacent_agents_PnA.append([i, self.perception_agent_num + j])
        # Action-Action adjacency
        for i in range(self.action_agent_num):
            for j in range(self.action_agent_num):
                if i != j:
                    pose1 = self.agent_state[self.perception_agent_num + i]
                    pose2 = self.agent_state[self.perception_agent_num + j]
                    if Agent_Util.adjacent_agents(pose1[0:self.agent_pose_dim-1], pose2[0:self.agent_pose_dim-1], hop_num=self.comm_hop):
                        self.adjacent_agents_AnA.append([self.perception_agent_num + i, self.perception_agent_num + j])

        # TODO: REWARD STRUCTURE #####################################################################################################################
        all_adjacencies = self.adjacent_agents_PnP + self.adjacent_agents_AnA + self.adjacent_agents_PnA
        if r_func == 'RF1':
            self.reward = self.get_reward1(len(self.onFire_List), sum(self.sensed_contribution), sum(self.pruned_contribution),
                                           all_adjacencies, time_passed)
        elif r_func == 'RF2':
            self.reward = self.get_reward2(len(self.onFire_List), sum(self.sensed_contribution), sum(self.pruned_contribution),
                                           all_adjacencies, time_passed)
        elif r_func == 'RF3':
            self.reward = self.get_reward3(len(self.onFire_List), sum(self.sensed_contribution), sum(self.pruned_contribution),
                                           all_adjacencies, time_passed)
        elif r_func is None:
            self.reward = self.get_reward1(len(self.onFire_List), sum(self.sensed_contribution), sum(self.pruned_contribution),
                                           all_adjacencies, time_passed)
        else:
            raise ValueError(">>> Oops! The specified Reward Function name doesn't exist. Options: RF1, RF2, RF3")


        # if all the fire fronts have been sensed, exit the environment
        # Perception performance: (sensed + pruned) / (active + pruned)
        self.perception_complete = (len(self.sensed_List) + len(self.pruned_List)) / (len(self.onFire_List) + len(self.pruned_List))
        # Action performance:  pruned / (active + pruned)
        self.action_complete = len(self.pruned_List) / (len(self.onFire_List) + len(self.pruned_List))
        # when more than 95% of firespots have been pruned, the agent wins the game
        # lower the bar to 80%
        if self.action_complete >= a_c_threshold:
            self.done = True
            self.reward += 1000.0

        return state, self.reward, self.done, self.perception_complete, self.action_complete

    # agents' state transisitions: updating individual agents' states [X, Y, Z] according to the taken action
    def agent_state_update(self, action_type, p_vel, a_vel, vert_vel=2, min_alt=5, max_alt=15):
        # Update P agents
        for i in range(self.perception_agent_num):
            # state transisitions
            # Action: Forward
            if action_type[i] == 0:
                self.agent_state[i][0] = max(self.agent_state[i][0] - p_vel, 0)
            # Action: Backward
            elif action_type[i] == 1:
                self.agent_state[i][0] = min(self.agent_state[i][0] + p_vel, self.world_size - 1)
            # Action: Left
            elif action_type[i] == 2:
                self.agent_state[i][1] = max(self.agent_state[i][1] - p_vel, 0)
            # Action: Right
            elif action_type[i] == 3:
                self.agent_state[i][1] = min(self.agent_state[i][1] + p_vel, self.world_size - 1)
            # Action: Up (altitude):
            elif action_type[i] == 4 and self.agent_state[i][3] == 0:
                self.agent_state[i][2] = min(self.agent_state[i][2] + vert_vel, max_alt)  # maximum allowed altitude set to 20
            # Action: Down (altitude):
            elif action_type[i] == 5 and self.agent_state[i][3] == 0:
                self.agent_state[i][2] = max(self.agent_state[i][2] - vert_vel, min_alt)  # minimum allowed altitude set to 5

        # Update A agents       
        for i in range(self.perception_agent_num, self.perception_agent_num + self.action_agent_num):
            if action_type[i] == 0:
                self.agent_state[i][0] = max(self.agent_state[i][0] - a_vel, 0)
            # Action: Backward
            elif action_type[i] == 1:
                self.agent_state[i][0] = min(self.agent_state[i][0] + a_vel, self.world_size - 1)
            # Action: Left
            elif action_type[i] == 2:
                self.agent_state[i][1] = max(self.agent_state[i][1] - a_vel, 0)
            # Action: Right
            elif action_type[i] == 3:
                self.agent_state[i][1] = min(self.agent_state[i][1] + a_vel, self.world_size - 1)
            elif action_type[i] == 4:
                continue

    # get the reward for agents (reward function number 1) - w/o time penalty, w/o firespot penalty, w/ communication reward
    def get_reward1(self, num_firespots, num_sensed, num_pruned, all_adjacencies, time_passed):
        # computing performance rewards
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
    def get_reward2(self, num_firespots, num_sensed, num_pruned, all_adjacencies, time_passed):
        #new_reward = self.old_reward_without_adjacent
        #new_reward = -1.0 * time_passed

        #im_reward = new_reward - self.old_reward_without_adjacent
        #self.old_reward_without_adjacent = new_reward
        if num_firespots > 0:
            im_reward = -1.0
        else:
            im_reward = 0.0

        return im_reward

    # get the reward for agents (reward function number 2) - w/o time penalty, w/ firespots penalty, w/o communication reward
    def get_reward3(self, num_firespots, num_sensed, num_pruned, all_adjacencies, time_passed):
        # computing performance rewards
        # new_reward = 2.0 * num_sensed
        # new_reward += 20.0 * num_pruned
        # new_reward = self.old_reward_without_adjacent
        # new_reward -= 0.1 * num_firespots  # small penalty for each active firespot in the field

        # im_reward = new_reward
        # self.old_reward_without_adjacent = new_reward
        im_reward = -0.1 * num_firespots

        return im_reward

    # generating the state matrix
    def state_gen(self):
        self.state = np.zeros((self.world_size, self.world_size), dtype=float)  # cleanup the previous states
        for i in range(len(self.agent_state)):
            # Perception Agents
            if self.agent_state[i][3] == 0:
                # mark the Perception agent scope
                for i1 in range(max(0, self.agent_state[i][0] - self.agent_state[i][2]),
                                min(self.agent_state[i][0] + self.agent_state[i][2] + 1, self.world_size)):
                    for j1 in range(max(0, self.agent_state[i][1] - self.agent_state[i][2]),
                                    min(self.agent_state[i][1] + self.agent_state[i][2] + 1, self.world_size)):
                        self.state[i1][j1] = 4  # Perception agent scope index
                # mark the position of the perception agents
                self.state[self.agent_state[i][0]][self.agent_state[i][1]] = 3  # Perception agent location index

            # Action Agents
            elif self.agent_state[i][3] == 1:
                # mark the Action agent scope
                for i1 in range(max(0, self.agent_state[i][0] - self.agent_state[i][2]),
                                min(self.agent_state[i][0] + self.agent_state[i][2] + 1, self.world_size)):
                    for j1 in range(max(0, self.agent_state[i][1] - self.agent_state[i][2]),
                                    min(self.agent_state[i][1] + self.agent_state[i][2] + 1, self.world_size)):
                        self.state[i1][j1] = 6  # Action agent scope index
                # mark the position of the action agents
                self.state[self.agent_state[i][0]][self.agent_state[i][1]] = 5  # Action agent location index
        # mark the sensed fire fronts
        for i in range(len(self.sensed_List)):
            self.state[self.sensed_List[i][0]][self.sensed_List[i][1]] = 1  # sensed firespots index
        # mark the pruned fire fronts
        for i in range(len(self.pruned_List)):
            self.state[self.pruned_List[i][0]][self.pruned_List[i][1]] = 2  # pruned firespots index

        return self.state

    # Get the FOV of each agent
    def get_FOV(self):
        """
        Get self.FOV_list
        If it is empty, then
            Get sensor image by cropping the self.state map
                Store as np.array in FOV_list
        Assume that the first N agents are always P agents
            N = self.perception_agent_num
        """
        if len(self.FOV_list) > 0:
            return self.FOV_list
        else:            
            FOV_list = []
            
            for i in range(self.perception_agent_num):
                # Perception Agents
                if self.agent_state[i][3] == 0:
                    #1. crop image
                    x, y, z, alt = self.agent_state[i]
                    # the upper-left corner of the agent searching scope
                    upper_x = max(0, x - z)
                    upper_y = max(0, y - z)
                    # the lower-right corner of the agent searching scope
                    lower_x = min(x + z, self.world_size - 1)
                    lower_y = min(y + z, self.world_size - 1)
                    # crop
                    cropped = self.state[upper_x:lower_x+1, upper_y:lower_y+1].copy()
                    
                    #2. change 2-6 to 0
                    cropped[cropped>=2] = 0
                    
                    #3. add to dict
                    FOV_list.append(cropped)
                else:
                    print('Idx Error!')
    
            return FOV_list


    # online visualization of the game state with PyGame
    def env_visualize(self):
        # fill the background screen with green
        self.screen.fill((197, 225, 165))
        # Agent_Util.on_Fire_Spot_Plot(self.screen, self.onFire_List)
        # plot the sensed fire with red dots
        Agent_Util.sensed_Fire_Spot_Plot(self.screen, self.sensed_List)
        Agent_Util.pruned_Fire_Spot_Plot(self.screen, self.pruned_List)

        for i in range(len(self.agent_state)):
            # Perception Agents
            if self.agent_state[i][3] == 0:
                # Calculate the size of the FOV scope
                searching_Scope_X = 2 * self.agent_state[i][2]
                searching_Scope_Y = 2 * self.agent_state[i][2]

                # the coordination of the upper-left corner of the agent searching scope
                searching_Scope_Upper_Left_Corner = (self.agent_state[i][0] - searching_Scope_X / 2, self.agent_state[i][1] - searching_Scope_Y / 2)
                # the size of the agent searching scope
                searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                # Plot the Perception agent (Circle) and its corresponding FOV scope (Rectangle)
                pygame.draw.circle(self.screen, (0, 0, 255), (int(self.agent_state[i][0]), int(self.agent_state[i][1])), 2)
                pygame.draw.rect(self.screen, (0, 0, 255), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

            elif self.agent_state[i][3] == 1:
                # the vertex set for the Action agent
                firefighter_Agent_Vertex = [
                    (self.agent_state[i][0] - 2, self.agent_state[i][1]),
                    (self.agent_state[i][0], self.agent_state[i][1] + 2),
                    (self.agent_state[i][0] + 2, self.agent_state[i][1]),
                    (self.agent_state[i][0], self.agent_state[i][1] - 2)]
                # plot the firefighter agent (Diamond)
                pygame.draw.polygon(self.screen, (128, 0, 128), firefighter_Agent_Vertex)

        # update the display according to the latest change
        pygame.display.update()

    # initialize the fire model
    def fire_init(self):
        # Fire region (Color: Red (255, 0, 0))
        # The wildfire generation and propagation utilizes the FARSITE wildfire mathematical model
        # To clarify the fire state data, the state of the fire spot at each moment is stored in the dictionary list separately
        # Besides, the current fire map will also be stored as a matrix with the same size of the simulation model, which
        # reflects the fire intensity of each position on the world

        # create the fire state dictionary list
        self.fire_States_List = []
        for i in range(self.fireAreas_Num):
            self.fire_States_List.append([])
        # length and width of the terrain as a list [length, width]
        terrain_sizes = [self.world_size, self.world_size]
        hotspot_areas = []
        for i in range(self.fireAreas_Num):
            hotspot_areas.append([self.fire_info[0][i][0] - 5, self.fire_info[0][i][0] + 5,
                                  self.fire_info[0][i][1] - 5, self.fire_info[0][i][1] + 5])

        # checking fire model setting mode and initializing the fire model
        if self.fire_info[1][9] == 0:  # when using "uniform" fire setting (all fire areas use the same parameters)
            # initial number of fire spots (ignition points) per hotspot area
            num_ign_points = self.fire_info[1][0]
            # fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
            fuel_coeff = self.fire_info[1][2]
            # average mid-flame wind velocity (higher values streches the fire more)
            wind_speed = self.fire_info[1][3]
            # wind azimuth
            wind_direction = np.pi * 2 * self.fire_info[1][4] / 360  # converting degree to radian

            # Init the wildfire model
            self.fire_mdl = WildFire(terrain_sizes=terrain_sizes, hotspot_areas=hotspot_areas, num_ign_points=num_ign_points, duration=self.duration,
                                     time_step=1, radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi / 3)
            self.ign_points_all = self.fire_mdl.hotspot_init()      # initializing hotspots
            self.fire_map = self.ign_points_all                     # initializing fire-map
            self.previous_terrain_map = self.ign_points_all.copy()  # initializing the starting terrain map
            self.geo_phys_info = self.fire_mdl.geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
                                                                  avg_wind_direction=wind_direction)  # initialize geo-physical info
        else:  # when using "Specific" fire setting (each fire area uses its own parameters)
            self.fire_mdl = []
            self.geo_phys_info = []
            self.ign_points_all = []
            self.previous_terrain_map = []
            self.new_fire_front_temp = []
            self.current_geo_phys_info = []
            # initialize fire areas separately
            for i in range(self.fireAreas_Num):
                self.new_fire_front_temp.append([])
                self.current_geo_phys_info.append([])
                # initial number of fire spots (ignition points) per hotspot area
                num_ign_points = self.fire_info[1][0][i]
                # fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
                fuel_coeff = self.fire_info[1][2][i]
                # average mid-flame wind velocity (higher values streches the fire more)
                wind_speed = self.fire_info[1][3][i]
                # wind azimuth
                wind_direction = np.pi * 2 * self.fire_info[1][4][i] / 360  # converting degree to radian

                # init the wildfire model
                self.fire_mdl.append(WildFire(
                    terrain_sizes=terrain_sizes, hotspot_areas=[hotspot_areas[i]], num_ign_points=num_ign_points, duration=self.duration, time_step=1,
                    radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi / 3))
                self.ign_points_all.append(self.fire_mdl[i].hotspot_init())        # initializing hotspots
                self.previous_terrain_map.append(self.fire_mdl[i].hotspot_init())  # initializing the starting terrain map
                self.geo_phys_info.append(self.fire_mdl[i].geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
                                                                              avg_wind_direction=wind_direction))  # initialize geo-physical info
            # initializing the fire-map
            self.fire_map = []
            for i in range(self.fireAreas_Num):
                for j in range(len(self.ign_points_all[i])):
                    self.fire_map.append(self.ign_points_all[i][j])
            self.fire_map = np.array(self.fire_map)
            self.fire_map_spec = self.ign_points_all

        # the lists to store the firespots in different state, coordinates only
        self.onFire_List = []  # the onFire_List, store the points currently on fire (sensed points included, pruned points excluded)
        self.sensed_List = []  # the sensed_List, store the points currently on fire and have been sensed by agents
        self.pruned_List = []  # the pruned_List, store the pruned fire spots

        # keeping track of agents' contributions (e.g. number of sensed/pruned firespot by each Perception/Action agent)
        self.sensed_contribution = [0] * self.perception_agent_num
        self.pruned_contribution = [0] * self.action_agent_num

    # propagate fire one step forward according to the fire model
    def fire_propagation(self):
        # checking fire model setting mode and initializing the fire model
        if self.fire_info[1][9] == 0:  # when using "uniform" fire setting (all fire areas use the same parameters)
            self.new_fire_front, current_geo_phys_info =\
                self.fire_mdl.fire_propagation(self.world_size, ign_points_all=self.ign_points_all, geo_phys_info=self.geo_phys_info,
                                               previous_terrain_map=self.previous_terrain_map, pruned_List=self.pruned_List)
            updated_terrain_map = self.previous_terrain_map
        else:  # when using "Specific" fire setting (each fire area uses its own parameters)
            updated_terrain_map = self.previous_terrain_map
            for i in range(self.fireAreas_Num):
                self.new_fire_front_temp[i], self.current_geo_phys_info[i] =\
                    self.fire_mdl[i].fire_propagation(self.world_size, ign_points_all=self.ign_points_all[i], geo_phys_info=self.geo_phys_info[i],
                                                      previous_terrain_map=self.previous_terrain_map[i], pruned_List=self.pruned_List)

            # update the new firefront list by combining all region-wise firefronts
            self.new_fire_front = []
            for i in range(self.fireAreas_Num):
                for j in range(len(self.new_fire_front_temp[i])):
                    self.new_fire_front.append(self.new_fire_front_temp[i][j])
            self.new_fire_front = np.array(self.new_fire_front)

        # update the region-wise fire map
        if self.fire_info[1][9] == 1:
            for i in range(self.fireAreas_Num):
                self.fire_map_spec[i] = np.concatenate([self.fire_map_spec[i], self.new_fire_front_temp[i]], axis=0)
        else:
            self.fire_map_spec = self.fire_map

        # process the fire spot information and generate the onFire and targer onfire list
        self.onFire_List, self.target_onFire_list = Agent_Util.fire_Data_Storage(
            self.new_fire_front, self.world_size, self.onFire_List, self.pruned_List, self.target_onFire_list, self.target_info)

        # updating the fire-map data for next step
        if self.new_fire_front.shape[0] > 0:
            self.fire_map = np.concatenate([self.fire_map, self.new_fire_front], axis=0)  # raw fire map without fire decay

        # update the fire propagation information
        if self.fire_info[1][9] == 1:
            ign_points_all_temp = []
            for i in range(self.fireAreas_Num):
                if self.new_fire_front_temp[i].shape[0] > 0:
                    # fire map with fire decay
                    self.previous_terrain_map[i] = np.concatenate((updated_terrain_map[i], self.new_fire_front_temp[i]), axis=0)
                ign_points_all_temp.append(self.new_fire_front_temp[i])
            self.ign_points_all = ign_points_all_temp
        else:
            if self.new_fire_front.shape[0] > 0:
                self.previous_terrain_map = np.concatenate((updated_terrain_map, self.new_fire_front))  # fire map with fire decay
                self.ign_points_all = self.new_fire_front

    # close pygame (only for online visualization option)
    @staticmethod
    def env_close():
        pygame.quit()

    # initialize the frame stack by adding some random action to it, the agents will take 4 random actions and fire will propagate from several
    # episodes, while the initialize episode number will not be considered in the formal training
    def env_init_stack(self, planar_vel, vert_vel, min_alt, max_alt):
        self.env_init()
        for _ in range(self.stack_num):
            action_p = np.random.randint(0, 6, 2)  # Perception agent action generator, using randint now
            action_a = np.random.randint(0, 4, 2)  # Action agent action generator, using randint now

            actions = list(action_p) + list(action_a)

            state, reward, done, perception_complete, action_complete = self.env_step(actions, planar_vel, vert_vel, min_alt, max_alt)  # step forward

        return self.frame

    # video_replay reconstruction information Buffer
    def env_replay_store(self):
        self.agent_state_buffer.append(self.agent_state)  # agent State
        self.sensed_List_Buffer.append(self.sensed_List)  # sensing List Buffer
        self.pruned_List_Buffer.append(self.pruned_List)  # pruning List Buffer
        self.reward_buffer.append(self.reward)            # Reward variations


if __name__ == '__main__':
    # initialize the env
    env = FireCommanderExtreme(online_vis=True)
    env.env_init()
    
    # go through episodes of the game
    num_episode = 1000
    for step in range(num_episode):
        action_p = np.random.randint(0, 6, 2)  # Perception agent action generator, using randint now
        action_a = np.random.randint(0, 5, 2)  # Action agent action generator, using randint now

        actions = list(action_p) + list(action_a)
    
        state, reward, done, perception_complete, action_complete = env.env_step(actions, vert_vel=2, min_alt=5, max_alt=15, time_passed=step,
                                                                                 a_c_threshold=1.1, r_func='RF2')

        print(step)
        env.env_visualize()  # env visualizer, for debugging
    
    # Visualize the last state
    print(reward)
    plt.imshow(state)
    plt.show()
    
    env.env_close()