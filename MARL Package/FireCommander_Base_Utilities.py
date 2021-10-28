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


# Environment utilities
class EnvUtilities(object):
    # determining the neighboring agents (discrete) and returning a binary flag at each time step (works with both 2D and 3D positions)
    @staticmethod
    def adjacent_agents(agent1_pose, agent2_pose, hop_num=1):
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
    @staticmethod
    def sensing(world_state, agent_loci, world_size, vision, state_space, feat_dim, NumP, NumA):
        # sensing
        world_state[world_state >= 2] = 0  # only taking into account the firespots that have not been put out yet
        temp = np.multiply(world_state, agent_loci)  # lay over the respective tensor layers to get the newly detected firespots
        sensed_list = np.where(temp.reshape(1, world_size * world_size) > 0)[1]  # form the sensed list

        # extracting 1-hot encoded FOV tensors
        FOV = EnvUtilities.FOV_encoding(state_space, agent_loci, feat_dim, NumP, NumA, world_size, vision)

        return sensed_list, FOV

    # Pruning the firemap (by Action agents)
    @staticmethod
    def pruning(world_state, agent_loci, sensed_list, firespot_loci, world_size):
        # pruning
        world_state[world_state != 2] = 0  # only taking into account the firespots that have been found
        temp = np.multiply(world_state, agent_loci)  # lay over the respective tensor layers to get the newly pruned firespots
        pruned_list = np.where(temp.reshape(1, world_size * world_size) > 0)[1]  # form the sensed list

        # removing pruned spots from sensed and firespot lists
        for idx in pruned_list:
            if idx in sensed_list:
                sensed_list = np.delete(sensed_list, np.where(sensed_list == idx))
            if idx in firespot_loci:
                firespot_loci = np.delete(firespot_loci, np.where(firespot_loci == idx))

        return pruned_list, sensed_list, firespot_loci

    # FOV encoding (i.e., 1-hot encoding from 2D matrix view into N-D tensor data)
    @staticmethod
    def FOV_encoding(state_space, agent_loci, feat_dim, NumP, NumA, world_size, vision):
        env_space = state_space[0].copy()  # building a temporary info map for FOV extraction
        # the agent view
        coords = np.argwhere(agent_loci > 0)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        agent_view = env_space[x_min:x_max + 1, y_min:y_max + 1]

        # padding the agent view if at edge
        m1, m2 = agent_view.shape
        if vision == 1:
            FOV = np.zeros(shape=(feat_dim, 3, 3))
            if (m1 < 3) or (m2 < 3):
                agent_view = EnvUtilities.padding_FOV_1hop(agent_view, agent_loci, world_size, -1)
        elif vision == 2:
            FOV = np.zeros(shape=(feat_dim, 5, 5))
            if (m1 < 5) or (m2 < 5):
                agent_view = EnvUtilities.padding_FOV_2hop(agent_view, agent_loci, world_size, -1)
        else:
            raise ValueError(">>> Oops! Sorry, for now, only 1-hop and 2-hop visions are available options...")

        # extracting/encoding the FOV information
        FOV_temp = np.zeros(shape=(feat_dim, 10, 10))
        FOV_temp[1][np.where(state_space[0] == 1)] = 1  # encoding the fire information (0 -> nothing, 1 -> fire not seen before)
        FOV_temp[2][np.where(state_space[0] == 2)] = 1  # encoding the fire information (0 -> nothing, 1 -> fire seen before)
        FOV_temp[3][np.where(state_space[0] == 3)] = 1  # encoding the fire information (0 -> nothing, 1 -> fire pruned before)
        for i in range(NumP):
            FOV_temp[4 + i][np.where(state_space[i + 1] == 1)] = 1  # encoding other P agent's locations
            FOV_temp[4 + NumP + i][np.where(state_space[i + 1] == 2)] = 1  # encoding other P agent's FOVs
        for i in range(NumA):
            FOV_temp[4 + (2 * NumP) + i][np.where(state_space[i + NumP + 1] == 1)] = 1  # encoding A agent's locations

        # cropping the FOV
        for i in range(feat_dim):
            temp = FOV_temp[i][x_min:x_max + 1, y_min:y_max + 1]
            if temp.shape != FOV[i].shape:
                if vision == 1:
                    if (m1 < 3) or (m2 < 3):
                        temp = EnvUtilities.padding_FOV_1hop(temp, agent_loci, world_size, 0)
                elif vision == 2:
                    if (m1 < 5) or (m2 < 5):
                        temp = EnvUtilities.padding_FOV_2hop(temp, agent_loci, world_size, 0)
                else:
                    raise ValueError(">>> Oops! Sorry, for now, only 1-hop and 2-hop visions are available options...")
                FOV[i] = temp
            else:
                FOV[i] = temp
        FOV[0][np.where(agent_view == -1)] = 1  # encoding the edge and padding information

        return FOV

    @staticmethod
    def FOV_vectorized_encoding(state_space, agent_loci, feat_dim, NumP, NumA, world_size, vision):
        # the agent view
        center = np.argwhere(agent_loci == 1)[0]
        x_min, y_min = center - vision
        x_max, y_max = center + vision

        # tiles in grid + number of agents + is out of bounds + if seen before
        agent_start_idx = world_size**2
        num_agents = NumP + NumA + 1
        out_bounds_idx = agent_start_idx + num_agents
        if_seen_idx = out_bounds_idx + 1

        square_size = if_seen_idx + 1

        width = 2 * vision + 1
        FOV = np.zeros((width, width, square_size))

        for i_x, x in enumerate(range(x_min, x_max + 1)):
            for i_y, y in enumerate(range(y_min, y_max + 1)):
                square = np.zeros(square_size)

                # out of bounds
                if x < 0 or y < 0 or x >= world_size or y >= world_size:
                    square[out_bounds_idx] = 1
                else:
                    # square location
                    square[:agent_start_idx] = EnvUtilities.position_one_hot(x, y, world_size)
                    
                    # agent type
                    for agent_idx in range(num_agents):
                        if agent_idx == 0:  # fire
                            agent_on_square = 1 if state_space[agent_idx][x, y] in [1, 2] else 0

                            # seen before
                            if state_space[agent_idx][x, y] == 2:
                                square[if_seen_idx] = 1
                        else:   # perception or action agent
                            agent_on_square = 1 if state_space[agent_idx][x, y] == 1 else 0

                        square[agent_start_idx + agent_idx] = agent_on_square

                FOV[i_x][i_y] = square

        return FOV

    # padding the FOV while at edges
    @staticmethod
    def padding_FOV_1hop(FOV, agent_loci, world_size, pad_val):
        m1, m2 = FOV.shape

        # top and bottom edges
        if m2 == 3:
            if m1 == 2:
                if np.where(agent_loci != 0)[0][0] > world_size/2:
                    FOV = np.pad(FOV, [(0, 1), (0, 0)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(1, 0), (0, 0)], mode='constant', constant_values=pad_val)

        # left and right edges
        if m1 == 3:
            if m2 == 2:
                if np.where(agent_loci != 0)[1][0] > world_size/2:
                    FOV = np.pad(FOV, [(0, 0), (0, 1)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(0, 0), (1, 0)], mode='constant', constant_values=pad_val)

        # the four corners
        if (m1 == 2) and (m2 == 2):
            if (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size/2):
                FOV = np.pad(FOV, [(0, 1), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size/2):
                FOV = np.pad(FOV, [(0, 1), (1, 0)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size/2):
                FOV = np.pad(FOV, [(1, 0), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size/2):
                FOV = np.pad(FOV, [(1, 0), (1, 0)], mode='constant', constant_values=pad_val)

        return FOV

    # padding the FOV while at edges
    @staticmethod
    def padding_FOV_2hop(FOV, agent_loci, world_size, pad_val):
        m1, m2 = FOV.shape

        # top and bottom edges
        if m2 == 5:
            if m1 == 4:
                if np.where(agent_loci != 0)[0][0] > world_size / 2:
                    FOV = np.pad(FOV, [(0, 1), (0, 0)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(1, 0), (0, 0)], mode='constant', constant_values=pad_val)
            if m1 == 3:
                if np.where(agent_loci != 0)[0][0] > world_size / 2:
                    FOV = np.pad(FOV, [(0, 2), (0, 0)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(2, 0), (0, 0)], mode='constant', constant_values=pad_val)

        # left and right edges
        if m1 == 5:
            if m2 == 4:
                if np.where(agent_loci != 0)[1][0] > world_size / 2:
                    FOV = np.pad(FOV, [(0, 0), (0, 1)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(0, 0), (1, 0)], mode='constant', constant_values=pad_val)
            if m2 == 3:
                if np.where(agent_loci != 0)[1][0] > world_size / 2:
                    FOV = np.pad(FOV, [(0, 0), (0, 2)], mode='constant', constant_values=pad_val)
                else:
                    FOV = np.pad(FOV, [(0, 0), (2, 0)], mode='constant', constant_values=pad_val)

        # the four corners
        if (m1 == 4) and (m2 == 4):
            if (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(0, 1), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(0, 1), (1, 0)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(1, 0), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(1, 0), (1, 0)], mode='constant', constant_values=pad_val)
        elif (m1 == 3) and (m2 == 3):
            if (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(0, 2), (0, 2)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(0, 2), (2, 0)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(2, 0), (0, 2)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(2, 0), (2, 0)], mode='constant', constant_values=pad_val)
        elif (m1 == 4) and (m2 == 3):
            if (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(0, 1), (0, 2)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(0, 1), (2, 0)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(1, 0), (0, 2)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(1, 0), (2, 0)], mode='constant', constant_values=pad_val)
        elif (m1 == 3) and (m2 == 4):
            if (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(0, 2), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] > world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(0, 2), (1, 0)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] > world_size / 2):
                FOV = np.pad(FOV, [(2, 0), (0, 1)], mode='constant', constant_values=pad_val)
            elif (np.where(agent_loci != 0)[0][0] < world_size / 2) and (np.where(agent_loci != 0)[1][0] < world_size / 2):
                FOV = np.pad(FOV, [(2, 0), (1, 0)], mode='constant', constant_values=pad_val)

        return FOV

    # turn x, y world coordinate into one hot array
    @staticmethod
    def position_one_hot(x, y, world_size):
        one_hot = np.zeros(world_size**2)
        one_hot[y * world_size + x] = 1

        return one_hot
