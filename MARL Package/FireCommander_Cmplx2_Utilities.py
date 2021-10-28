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

# Import the PyGame package
import pygame

from pygame.locals import *
from sys import exit
import pickle
import numpy as np
# from Utilities import Utilities as Util
import matplotlib.pyplot as plt
import os, sys
import shutil
from WildFire_Model import WildFire
import time


# Environment utilities
class EnvUtilities(object):
    # The function to plot the target
    # Input value: current screen, target loci infomation list, target center postion (X, Y), target size (X, Y), the flag to plot the edge or not
    # Output value: the updated list
    @staticmethod
    def target_Plot(screen, target_info):
        for i in range(len(target_info)):
            for j in range(len(target_info[i])):
                # The coordination of the upper-left corner of target
                target_Upper_Left_Corner = (target_info[i][j][0] - 5, target_info[i][j][1] - 5)
                target_Size = (10, 10)
                # Plot the normaltarget, fill the rectangle with orange
                if target_info[i][j][2] == 0:
                    pygame.draw.rect(screen, (255, 165, 0), Rect(target_Upper_Left_Corner, target_Size))

                elif target_info[i][j][2] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), Rect(target_Upper_Left_Corner, target_Size))

    # The function fire_Data_Storage is used to separate the fire spots in different regions for storage
    # Input: The number of the ignited fire spots, the whole fire map, the generated fire spot, the size of the
    #         simulation environment, the number of the fire regions and the previous fire state list and fire map,
    #         the previous onFire_List
    # Output: The updated fire map, the fire state list for storage, the updated onFire_List
    @staticmethod
    def fire_Data_Storage(new_fire_front, world_Size, onFire_List, pruned_List, target_onFire_list, target_info):
        if new_fire_front.shape[0] > 0:
            # Write the fire spot into the current world map list
            for i in range(new_fire_front.shape[0]):
                # # Ensure that all the fire spots to be displayed must be within the window scope
                if ((int(new_fire_front[i][0]) <= (world_Size - 1)) and (int(new_fire_front[i][1]) <= (world_Size - 1))
                        and (int(new_fire_front[i][0]) >= 0) and (int(new_fire_front[i][1]) >= 0)):

                    # If the new fire front points is not included in the current onFire_List, add it into the list
                    if ([int(new_fire_front[i][0]), int(new_fire_front[i][1])] not in onFire_List) and \
                            ([int(new_fire_front[i][0]), int(new_fire_front[i][1])] not in pruned_List):
                        onFire_List.append([int(new_fire_front[i][0]), int(new_fire_front[i][1])])

                        # Determine whether the new fire fronts locate inside the target region
                        for i1 in range(len(target_onFire_list)):
                            for j1 in range(len(target_onFire_list[i1])):
                                if (int(new_fire_front[i][0]) > (target_info[i1][j1][0] - 5)) and (int(new_fire_front[i][0]) < (target_info[i1][j1][0] + 5)) and \
                                (int(new_fire_front[i][1]) > (target_info[i1][j1][1] - 5)) and (int(new_fire_front[i][1]) < (target_info[i1][j1][1] + 5)):
                                    target_onFire_list[i1][j1] += 1

        return onFire_List, target_onFire_list

    # determining the neighboring agents (discrete) and returning a binary flag at each time step (works with both 2D and 3D positions)
    @staticmethod
    def adjacent_agents(agent1_pose, agent2_pose, hop_num=1):
        """
        determining the neighboring agents (discrete) and returning the indexes of neighbors at each time step

        :param agent1_pose: first agents position
        :param agent2_pose: second agents position
        :param hop_num: number of hops (discrete) for communication range (default:: 1-hop)
        :return: binary flag
        """

        if len(agent1_pose) != len(agent2_pose):
            raise ValueError(">>> Oops! Agent coordinates must be in same dimensions (either 2D or 3D).")

        adjacent_agents = False
        if len(agent1_pose) == 2:
            x1, x2, y1, y2 = agent1_pose[0], agent2_pose[0], agent1_pose[1], agent2_pose[1]
            if (abs(x1 - x2) <= hop_num) and (abs(y1 - y2) <= hop_num):
                adjacent_agents = True
        elif len(agent1_pose) == 3:
            x1, x2, y1, y2, z1, z2 = agent1_pose[0], agent2_pose[0], agent1_pose[1], agent2_pose[1], agent1_pose[2], agent2_pose[2]
            if (abs(x1 - x2) <= hop_num) and (abs(y1 - y2) <= hop_num) and (abs(z1 - z2) <= hop_num):
                adjacent_agents = True
        else:
            raise ValueError(">>> Oops! Agent coordinates must be in either 2D or 3D.")

        return adjacent_agents

    # Sense the firemap (by Perception agents)
    '''
    Newly added: return the FOV of selected agent
    '''
    @staticmethod
    def fire_Sensing(onFire_List, agent_loci, sensed_List, world_size):
        # Input: fire_Map, current agent state, the agent's FOV, geometric_physics info, sensed_List, window size
        # Output: fire sensed map (for agent status recording), CoM info, the list of the coordinates of the sensed points

        confidence_level = 1 - 3 / 50 * (agent_loci[2] - 10) # the sensing confidence level (highest when altitude=minimu_allowed and vice versa)

        onFire_List = np.array(onFire_List)

        # The coordination of the upper-left corner of the agent searching scope
        (tl_x, tl_y) = (agent_loci[0] - agent_loci[2], agent_loci[1] - agent_loci[2])

        # The coordination of the lower-right corner of the agent searching scope
        (br_x, br_y) = (agent_loci[0] + agent_loci[2], agent_loci[1] + agent_loci[2])

        # Search for the current fire map, determine whether the given fire spot locates within the searching scope
        if onFire_List.shape[0] > 0:
            raw_sensed_idx = np.intersect1d(np.argwhere(onFire_List[:, 0] <= min(br_x, world_size - 1)), np.argwhere(onFire_List[:, 0] >= max(tl_x, 0)))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(onFire_List[:, 1] <= min(br_y, world_size - 1)))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(onFire_List[:, 1] >= max(tl_y, 0)))
            # Apply the stochastic perception
            raw_sensed_idx = np.random.choice(raw_sensed_idx, int(round(confidence_level * len(raw_sensed_idx))), replace=False)

            '''
            Newly added: get the FOV of this agent
            '''
            upper_x = max(0, tl_x)
            upper_y = max(0, tl_y)

            lower_x = min(br_x, world_size - 1)
            lower_y = min(br_y, world_size - 1)

            size_x = lower_x - upper_x + 1
            size_y = lower_y - upper_y + 1

            FOV = np.zeros((size_x, size_y), dtype=float)

            for i in range(len(raw_sensed_idx)):
                tmp_x, tmp_y = onFire_List[raw_sensed_idx[i]]
                FOV[tmp_x - upper_x, tmp_y - upper_y] = 1

            # Remove the duplicate fire fronts
            raw_sensed_list = onFire_List[raw_sensed_idx, :]
            if len(raw_sensed_list[:, 0]) > 0:
                int_sensed_list = np.zeros((len(raw_sensed_list[:, 0]), 2), dtype=int)
                int_sensed_list[:, 0] = raw_sensed_list[:, 0].astype(int)
                int_sensed_list[:, 1] = raw_sensed_list[:, 1].astype(int)
                uni_int_sensed_list = np.unique(int_sensed_list, axis=0)

                sensed_List_copy = sensed_List.copy()
                if len(sensed_List_copy) > 0:
                    sensed_List_copy[0:0] = list(uni_int_sensed_list).copy()
                    sensed_List = np.unique(np.array(sensed_List_copy), axis=0).tolist()
                else:
                    sensed_List = uni_int_sensed_list.copy().tolist()
        elif onFire_List.shape[0] == 0:
            '''
            Newly added: get the FOV of this agent
            '''
            upper_x = max(0, tl_x)
            upper_y = max(0, tl_y)

            lower_x = min(br_x, world_size - 1)
            lower_y = min(br_y, world_size - 1)

            size_x = lower_x - upper_x + 1
            size_y = lower_y - upper_y + 1

            FOV = np.zeros((size_x, size_y), dtype=float)

        return sensed_List, FOV

    # Pruning the fire with the Action agents, if given fire spots locate within the firefighter agents' scope,
    # delete them from the onFire and sensed list, add them into the pruned list, create the pruned list with time stamp
    # for the current agent (For data storage)
    @staticmethod
    def fire_Pruning(agent_loci, onFire_List_raw, sensed_List, pruned_List,
                     new_fire_front, target_onFire_list, target_info, world_size, confidence_level):
        # Input: fire_Map, current agent state, the agent's FOV, onFire_List, sensed_List, pruned_List, new fire front list
        # Output: fire_Pruned_Map (for agent status recording), the updated fire_map, onFire_List, sensed_List, pruned_List

        # The coordination of the upper-left corner of the agent searching scope
        (tl_x, tl_y) = (agent_loci[0] - agent_loci[2], agent_loci[1] - agent_loci[2])

        # The coordination of the lower-right corner of the agent searching scope
        (br_x, br_y) = (agent_loci[0] + agent_loci[2], agent_loci[1] + agent_loci[2])

        onFire_List = np.array(onFire_List_raw)

        # sensed list flag, if there is any points that is included in the sensed list, this flag will become 1
        sensed_flag = 0

        # Search for the current fire map, determine whether the given fire spot locates within the searching scope
        if onFire_List.shape[0] > 0:
            raw_sensed_idx = np.intersect1d(np.argwhere(onFire_List[:, 0] <= min(br_x, world_size - 1)), np.argwhere(onFire_List[:, 0] >= max(tl_x, 0)))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(onFire_List[:, 1] <= min(br_y, world_size - 1)))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(onFire_List[:, 1] >= max(tl_y, 0)))
            # Apply the stochastic pruning
            raw_sensed_idx = np.random.choice(raw_sensed_idx, int(round(confidence_level * len(raw_sensed_idx))),
                                              replace=False)
            # Temporary list to store the fire front that may be pruned
            temp_list = onFire_List[raw_sensed_idx, :]
            if len(temp_list) > 0:
                int_pruned_list = temp_list.copy()
                uni_int_pruned_list = np.unique(int_pruned_list, axis=0)

                # Determine whether there's any sensed points within the scope
                sensed_List_copy = sensed_List.copy()

                if len(sensed_List_copy) > 0:
                    sensed_List_copy[0:0] = list(uni_int_pruned_list).copy()
                    if len(np.unique(np.array(sensed_List_copy), axis=0).tolist()) < (len(sensed_List) + len(uni_int_pruned_list)):
                        sensed_flag = 1

                # The pruning agent could only put out fire region that contains the sensed fire fronts
                if sensed_flag == 1:
                    for i in range(len(temp_list)):
                        # If the new fire front points is not included in the current onFire_List, add it into the list
                        if (([int(temp_list[i][0]), int(temp_list[i][1])] not in pruned_List) and
                                ([int(temp_list[i][0]), int(temp_list[i][1])] in onFire_List)):
                            onFire_List_raw.remove([int(temp_list[i][0]), int(temp_list[i][1])])
                            if ([int(temp_list[i][0]), int(temp_list[i][1])] in sensed_List):
                                sensed_List.remove([int(temp_list[i][0]), int(temp_list[i][1])])

                            # Determine whether the new fire fronts locate inside the target region
                            for i1 in range(len(target_onFire_list)):
                                for j1 in range(len(target_onFire_list[i1])):
                                    if (int(temp_list[i][0]) > (target_info[i1][j1][0] - 5)) and (int(temp_list[i][0]) < (target_info[i1][j1][0] + 5)) and \
                                    (int(temp_list[i][1]) > (target_info[i1][j1][1] - 5)) and (int(temp_list[i][1]) < (target_info[i1][j1][1] + 5)):
                                        target_onFire_list[i1][j1] -= 1

                            pruned_List.append([int(temp_list[i][0]), int(temp_list[i][1])])
        # elif onFire_List.shape[0] == 0:
        #     print('>>> Same bug, second function but I am ignoring it...')

        return onFire_List_raw, sensed_List, pruned_List, new_fire_front, target_onFire_list

    # This function intends to plot the sensed fire spot on the screen
    # Input: current screen, sensed fire spot list
    @staticmethod
    def sensed_Fire_Spot_Plot(screen, sensed_List):
        # Search for all the sensing agents' data
        for i in range(len(sensed_List)):
            # Plot the fire spot using the red color the corresponds to the intensity
            pygame.draw.circle(screen, (255, 0, 0),(int(sensed_List[i][0]), int(sensed_List[i][1])), 1)

    # This function intends to plot the pruned fire spot on the screen
    # Input: current screen, pruned fire spot list
    @staticmethod
    def pruned_Fire_Spot_Plot(screen, pruned_List):
        # Search for all the sensing agents' data
        for i in range(len(pruned_List)):
            # Plot the fire spot using the red color the corresponds to the intensity
            pygame.draw.circle(screen, (0, 0, 0),(int(pruned_List[i][0]), int(pruned_List[i][1])), 1)

    # This function intends to plot the onFire fire spot on the screen
    # Input: current screen, onFire fire spot list
    @staticmethod
    def on_Fire_Spot_Plot(screen, sensed_List):
        # Search for all the sensing agents' data
        for i in range(len(sensed_List)):
            # Plot the fire spot using the red color the corresponds to the intensity
            pygame.draw.circle(screen, (0, 255, 255),(int(sensed_List[i][0]), int(sensed_List[i][1])), 1)
