"""
# *******************************<><><><><>************************************
# *  FireCommander 2020 - An Interactive Joint Perception-Action Environment  *
# *******************************<><><><><>************************************
#
# Properties of CORE Robotics Lab
#	- Institute for Robotics & Intelligent Machines (IRIM), Georgia Institute
#		of Technology, Atlanta, GA, United States, 30332
#
# Authors
#	- Esmaeil Seraj* <IRIM, School of ECE, Georgia Tech - eseraj3@gatech.edu>
#	- Xiyang Wu <School of ECE, Georgia Tech - xwu391@gatech.edu>
#	- Matthew Gombolay (Ph.D) <IRIM, School of IC, Georgia Tech>
#
#	- *Esmaeil Seraj >> Author to whom any correspondences shall be forwarded
#
# Dependencies and Tutorials
#	- GitHub: ................... https://github.com/EsiSeraj/FireCommander2020
#	- Documentation (arXiv): .................................. [Add_Link_Here]
#	- PPT Tutorial: ........................................... [Add_Link_Here]
#	- Video Tutorial: ............................ https://youtu.be/UQsWPh9c3eM
#	- Supported by Python 3.6.4 and PyGame 1.9.6 (or any later version)
#
# Licence
# - (C) CORE Robotics Lab. All Rights Reserved - FireCommander 2020 (TM)
#
# - <FireCommander 2020 - An Interactive Joint Perception-Action Robotics Game>
#	Copyright (C) <2020> <Esmaeil Seraj, Xiyang Wu and Matthew C. Gombolay>
#
#	This program is free software; you can redistribute it and/or modify it
# 	under the terms of the GNU General Public License as published by the
# 	Free Software Foundation; either version 3.0 of the License, or (at your
# 	option) any later version.
#
# 	This program is distributed in the hope that it will be useful, but
# 	WITHOUT ANY WARRANTY; without even the implied warranty of
# 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# 	Public License for more details. 
#
#	You should have received a copy of the
# 	GNU General Public License along with this program; if not, write to the
# 	Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# 	MA  02110-1301, USA.
#
"""


# Import the PyGame package
import pygame

from pygame.locals import *

# wildfire simulation
class Animation_Reconstruction_Reconn_Utilities(object):
    # The function to plot the target
    # Input value: current screen, target loci infomation list, target center postion (X, Y), target size (X, Y), the flag to plot the edge or not
    # Output value: the updated list
    def target_Plot(self, screen, hospital_Font, target_Loci_Current):
        # The coordination of the upper-left corner of target
        target_Upper_Left_Corner = (target_Loci_Current[0] - target_Loci_Current[2] / 2,
                                    target_Loci_Current[1] - target_Loci_Current[3] / 2)
        target_Size = (target_Loci_Current[2], target_Loci_Current[3])
        # Plot the normaltarget, fill the rectangle with orange
        if target_Loci_Current[4] == 0:
            pygame.draw.rect(screen, (255, 165, 0), Rect(target_Upper_Left_Corner, target_Size))
            # The vertex set for the target's roof
            firefighter_Agent_Vertex = [(target_Loci_Current[0] - target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0] + target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0], target_Loci_Current[1] - target_Loci_Current[3])]
            # Plot the target's roof (Triangle)
            pygame.draw.polygon(screen, (210, 105, 30), firefighter_Agent_Vertex)
            # Display the goal on the screen
            screen.blit(hospital_Font.render('A', False, (0, 0, 0)), (target_Loci_Current[0] - 10,
                                                                         target_Loci_Current[1] - 20))

        elif target_Loci_Current[4] == 1:
            pygame.draw.rect(screen, (255, 255, 255), Rect(target_Upper_Left_Corner, target_Size))
            # The vertex set for the target's roof
            firefighter_Agent_Vertex = [(target_Loci_Current[0] - target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0] + target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0], target_Loci_Current[1] - target_Loci_Current[3])]
            # Plot the target's roof (Triangle)
            pygame.draw.polygon(screen, (210, 105, 30), firefighter_Agent_Vertex)
            # Display the goal on the screen
            screen.blit(hospital_Font.render('H', False, (255, 0, 0)), (target_Loci_Current[0] - 10,
                                                                        target_Loci_Current[1] - 20))

        elif target_Loci_Current[4] == 2:
            pygame.draw.rect(screen, (65, 105, 225), Rect(target_Upper_Left_Corner, target_Size))
            # The vertex set for the target's roof
            firefighter_Agent_Vertex = [(target_Loci_Current[0] - target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0] + target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0], target_Loci_Current[1] - target_Loci_Current[3])]
            # Plot the target's roof (Triangle)
            pygame.draw.polygon(screen, (210, 105, 30), firefighter_Agent_Vertex)
            # Display the goal on the screen
            screen.blit(hospital_Font.render('P', False, (255, 255, 0)), (target_Loci_Current[0] - 10,
                                                                        target_Loci_Current[1] - 20))

        # If the edge of the target is enabled, plot the edge with line width 2
        if target_Loci_Current[5] == 1:
            pygame.draw.rect(screen, (0, 0, 0), Rect(target_Upper_Left_Corner, target_Size), 2)
            # The vertex set for the target's roof
            firefighter_Agent_Vertex = [(target_Loci_Current[0] - target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0] + target_Loci_Current[2] / 2,
                                         target_Loci_Current[1] - target_Loci_Current[3] / 2),
                                        (target_Loci_Current[0], target_Loci_Current[1] - target_Loci_Current[3])]
            # Plot the target's roof (Triangle)
            pygame.draw.polygon(screen, (0, 0, 0), firefighter_Agent_Vertex, 2)

    # The function to plot the agent base
    # Input value: current screen, agent base infomation list, the flag to plot the edge or not
    # Output value: the updated list
    def agent_Base_Plot(self, screen, agent_Base_Info):
        # The coordination of the upper-left corner of agent base
        agent_Base_Upper_Left_Corner = (
            agent_Base_Info[0] - agent_Base_Info[2] / 2, agent_Base_Info[1] - agent_Base_Info[3] / 2)
        agent_Base_Size = (agent_Base_Info[2], agent_Base_Info[3])
        # Plot the agent base, fill the rectangle with orange
        pygame.draw.rect(screen, (255, 225, 0), Rect(agent_Base_Upper_Left_Corner, agent_Base_Size))
        # If the edge of the agent base is enabled, plot the edge with line width 2
        if agent_Base_Info[5] == 1:
            pygame.draw.rect(screen, (0, 0, 0), Rect(agent_Base_Upper_Left_Corner, agent_Base_Size), 2)

    # The function fire_Data_Storage is used to separate the fire spots in different regions for storage
    # Input: The number of the ignited fire spots, the whole fire map, the generated fire spot, the size of the
    #         simulation environment, the number of the fire regions and the previous fire state list and fire map,
    #         the previous onFire_List
    # Output: The updated fire map, the fire state list for storage, the updated onFire_List
    def onFire_List_Recovery(self, num_ign_points, fire_States_List, world_Size, fireSpots_Num, onFire_List, time, spec_flag, set_time):
        new_fire_front = []
        # Write the fire spot into the current world map list
        if spec_flag == 0:
            for i in range(fireSpots_Num):
                for j in range(num_ign_points):
                    if (num_ign_points * time + j) < len(fire_States_List[i]):
                        if (set_time * 1000) < fire_States_List[i][num_ign_points * time + j][3]:
                            new_fire_front.append([fire_States_List[i][num_ign_points * time + j][0],
                                                   fire_States_List[i][num_ign_points * time + j][1],
                                                   fire_States_List[i][num_ign_points * time + j][2]])
        else:
            for i in range(fireSpots_Num):
                for j in range(num_ign_points[i]):
                    if (num_ign_points[i] * time + j) < len(fire_States_List[i]):
                        if ((set_time[i] * 1000) < fire_States_List[i][num_ign_points[i] * time + j][3]):
                            new_fire_front.append([fire_States_List[i][num_ign_points[i] * time + j][0],
                                                   fire_States_List[i][num_ign_points[i] * time + j][1],
                                                   fire_States_List[i][num_ign_points[i] * time + j][2]])

        # Write the fire spot into the current world map list
        for i in range(len(new_fire_front)):
            # # Ensure that all the fire spots to be displayed must be within the window scope
            if ((new_fire_front[i][0] <= (world_Size - 1)) and (new_fire_front[i][1] <= (world_Size - 1))
                    and (new_fire_front[i][0] >= 0) and (new_fire_front[i][1] >= 0)):

                # If the new fire front points is not included in the current onFire_List, add it into the list
                if ([int(new_fire_front[i][0]), int(new_fire_front[i][1])] not in onFire_List):
                    onFire_List.append([int(new_fire_front[i][0]), int(new_fire_front[i][1])])

        return new_fire_front, onFire_List

    # check if a point is inside FOV
    @staticmethod
    def in_fov(br_x=None, br_y=None, tl_x=None, tl_y=None, x=None, y=None):
        """
        this function checks if a specific point is inside the FOV of an UAV. The FOV is specified by two of its coordinates

        :param br_x: x bottom right of FOV
        :param br_y: y bottom right of FOV
        :param tl_x: x top left of FOV
        :param tl_y: y top left of FOV
        :param x: x of the point to be checked
        :param y: y of the point to be checked
        :return: boolean flag
        """
        if br_x is None or br_y is None or tl_x is None or tl_y is None or x is None or y is None:
            raise ValueError(">>> Oops! Function 'in_fov()' needs ALL of its input arguments to work!")

        if x >= tl_x and x <= br_x and y >= tl_y and y <= br_y:
            return True
        else:
            return False

    def pruned_List_Recovery(self, loaded_pruned_Fire_List, pruned_List, time):
        for i in range(len(loaded_pruned_Fire_List)):
            if len(loaded_pruned_Fire_List[i][time]) > 0:
                for j in range(len(loaded_pruned_Fire_List[i][time][0])):
                    pruned_List.append([int(loaded_pruned_Fire_List[i][time][0][j][0]), int(loaded_pruned_Fire_List[i][time][0][j][1])])
        return pruned_List

    # Determine whether the given point inside the base
    def in_Agent_Base_Region(self, goal_X, goal_Y, agent_Base_Num, agent_Base_Loci_Full):
        in_Base_Flag = False
        base_Index = -1
        for i in range(agent_Base_Num):
            agent_Base_Loci = agent_Base_Loci_Full[i][len(agent_Base_Loci_Full[i]) - 1]
            if ((goal_X >= (agent_Base_Loci[0] - agent_Base_Loci[0] / 2)) and
                (goal_X <= (agent_Base_Loci[0] + agent_Base_Loci[0] / 2)) and
                (goal_Y >= (agent_Base_Loci[1] - agent_Base_Loci[1] / 2)) and
                (goal_Y <= (agent_Base_Loci[1] + agent_Base_Loci[1] / 2))):
                in_Base_Flag = True
                base_Index = i
                break
        return in_Base_Flag, base_Index

    # This function intends to plot the sensed fire spot on the screen
    # Input: current screen, sensed fire spot list
    def sensed_Fire_Spot_Plot(self, screen, sensed_List, time, current_Max_Intensity):
        # Search for all the sensing agents' data
        for i in range(len(sensed_List)):
            if current_Max_Intensity < sensed_List[i][2]:
                current_Max_Intensity = sensed_List[i][2]
            # Plot the fire spot using the red color the corresponds to the intensity
            pygame.draw.circle(screen, (
            sensed_List[i][2] * 155 / current_Max_Intensity + 100, 0, 0),
                               (int(sensed_List[i][0]), int(sensed_List[i][1])), 1)
        return current_Max_Intensity

    # Plot the lake
    def lake_plot(self, screen, lake_list, time):
        for i in range(len(lake_list)):
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][time][0], lake_list[i][time][1]), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][time][0] + 80, lake_list[i][time][1] - 80), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][time][0] + 20, lake_list[i][time][1] + 80), 100)

    # Plot roads that connect each target
    def road_plot(self, screen, target_Loci):
        for i in range(len(target_Loci)):
            for j in range(i, len(target_Loci)):
                pygame.draw.line(screen, (139, 69, 19), (target_Loci[i][0][0], target_Loci[i][0][1]),
                                 (target_Loci[j][0][0], target_Loci[j][0][1]), 5)