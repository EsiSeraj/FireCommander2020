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


# This file is used to store the essential parameters for scenario setting (25 Scenarios in all)
# Scenario_para 0 - 23 refer to the Scenario #1 - #24 in the scenario mode, while Scenario_para 24 refers to the practice scenario
# The format of the scenario setting is:
# General Structure: [General environment setting, General agent setting, Specific environment setting, Specific agent setting]
# General environment setting (environment_para): The number of each objects in the scenario
# [World size, Duration, Fire spot number, House number, Hospital number, Power station number, Lake number]

# General agent setting (robo_team_para): The number and control mode of each kind of agents in the scenario
# [Perception agent number, Action agent number, Hybrid agent number, Control mode (Homogenous/Heterogenous)]

# Specific environment setting (set_Loci): The details of each objects in the scenario
# [[Fire Loci 1, Fire Loci 2, ...],
# Fire Setting List*,
# [Agent Base Loci (Unique), Horizontal/Vertical],
# [House Loci 1, House Loci 2, ...],
# [Hospital Loci 1, Hospital Loci 2, ...],
# [Power station Loci 1, Power station Loci 2, ...],
# [Lake Loci 1, Lake Loci 2, ...]]
# *Note: Fire Setting List is divided into two separate modes, Uniform (All fire spots share the same setting) and Specific (Each fire spot uses its own setting)
# Fire Setting List (Uniform): [Fire Spot Number, Fire delay time, Fuel coefficient, Wind speed, Wind direction, Temporal penalty coefficient, Fire propagation weight,
# Action Pruning Confidence Level (In percentage), Hybrid Pruning Confidence Level (In percentage), 0 (Mode flag)]
# Fire Setting List (Specific): [[Fire Spot Number 1, Fire Spot Number 2, ...], [Fire delay time 1, Fire delay time 2, ...], [Fuel coefficient 1, Fuel coefficient 2, ...],
# [Wind speed 1, Wind speed 2, ...], [Wind direction 1, Wind direction 2, ...], Temporal penalty coefficient, Fire propagation weight,
# Action Pruning Confidence Level (In percentage), Hybrid Pruning Confidence Level (In percentage), 1 (Mode flag)]

# Specific agent setting (adv_setting): The details of each kind of agent in the scenario
# [Perception agent altitude limit, Hybrid agent altitude limit, Perception agent battery limit, Action agent battery limit,
# Hybrid agent battery limit, Perception agent velocity limit, Action agent velocity limit, Hybrid agent velocity limit,
# Action water tank limit, Hybrid water tank limit]
# The general form of each kind of list (battery, velocity and tank limit) is:
# E.g. Perception agent battery limit: [Perception agent 1 battery limit, Perception agent 2 battery limit, ...]

class scenario_setting():
    def __init__(self):
        super(scenario_setting, self).__init__()
        self.scenario_para = []
        for i in range(25):
            self.scenario_para.append([])

        self.scenario_para[0] = [[1200, 180, 1, 5, 1, 1, 1], [2, 2, 0, 0], [[[350, 350]], [10, 0, 10, 5, 45, 1.25, 0.1, 90, 80, 0], [[100, 600], 1], [[900, 200], [1100, 200], [900, 400], [1100, 400], [1000, 600]], [[600, 1000]], [[300, 1000]], [[650, 300]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[1] = [[1200, 180, 1, 1, 1, 0, 1], [2, 1, 0, 0], [[[350, 350]], [[15], [60], [15], [3], [45], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[800, 300]], [[900, 700]], [], [[450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[2] = [[1200, 180, 1, 1, 1, 1, 1], [1, 1, 0, 0], [[[350, 350]], [5, 0, 15, 5, 45, 1.25, 0.1, 90, 80, 0], [[100, 600], 1], [[1000, 700]], [[700, 1000]], [[300, 900]], [[750, 200]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[3] = [[1200, 180, 1, 1, 1, 0, 1], [1, 1, 0, 0], [[[350, 350]], [[12], [60], [5], [3], [45], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[800, 300]], [[900, 700]], [], [[450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[4] = [[1200, 180, 2, 2, 1, 1, 1], [3, 2, 0, 0], [[[350, 350], [350, 350]], [[3, 8], [0, 0], [5, 10], [5, 3], [45, 75], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 700], [1100, 700]], [[300, 1000]], [[600, 1000]], [[650, 200]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[5] = [[1200, 180, 2, 2, 1, 1, 1], [2, 3, 0, 0], [[[350, 350], [350, 350]], [[5, 3], [60, 0], [10, 10], [5, 5], [45, 75], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[800, 300], [1000, 300]], [[900, 800]], [[700, 1000]], [[350, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[6] = [[1200, 180, 2, 2, 1, 1, 1], [3, 1, 0, 0], [[[350, 750], [1050, 1050]], [[3, 3], [0, 0], [5, 5], [5, 5], [135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 600], [1100, 600]], [[700, 1100]], [[300, 1100]], [[650, 300]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[7] = [[1200, 180, 2, 2, 1, 1, 1], [3, 1, 0, 0], [[[350, 750], [1050, 1050]], [[5, 7], [0, 60], [3, 3], [10, 10], [135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[700, 300], [900, 300]], [[900, 700]], [[200, 1000]], [[650, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[8] = [[1200, 180, 2, 2, 1, 1, 1], [2, 2, 0, 0], [[[350, 350], [350, 350]], [[5, 5], [0, 0], [10, 10], [5, 5], [45, 75], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[800, 700], [1000, 700]], [[600, 1000]], [[300, 1000]], [[650, 200]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[9] = [[1200, 180, 2, 2, 1, 1, 1], [2, 2, 0, 0], [[[350, 350], [350, 350]], [[5, 5], [60, 0], [10, 10], [5, 5], [45, 75], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[800, 300], [1000, 300]], [[900, 800]], [[700, 1000]], [[350, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[10] = [[1200, 180, 2, 2, 1, 1, 1], [2, 2, 0, 0], [[[350, 750], [1050, 1050]], [[5, 5], [0, 0], [5, 5], [5, 5], [135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 700], [1100, 700]], [[700, 1000]], [[300, 1000]], [[650, 200]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[11] = [[1200, 180, 2, 2, 1, 1, 1], [3, 2, 0, 0], [[[350, 750], [1050, 1050]], [[3, 10], [0, 60], [5, 10], [5, 10], [135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[700, 300], [900, 300]], [[900, 700]], [[200, 1000]], [[550, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[12] = [[1200, 180, 3, 5, 2, 1, 0], [2, 2, 0, 0], [[[350, 450], [350, 750], [1050, 1050]], [[3, 3, 3], [0, 60, 0], [10, 10, 10], [5, 5, 5], [75, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 200], [1100, 200], [900, 400], [1100, 400], [1000, 600]], [[600, 200], [300, 1100]], [[600, 1000]], []], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[13] = [[1200, 180, 3, 2, 1, 1, 1], [3, 2, 0, 0], [[[350, 750], [1050, 150], [1050, 1050]], [[5, 5, 5], [0, 120, 60], [10, 10, 10], [5, 5, 5], [135, 315, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[600, 200], [800, 200]], [[900, 700]], [[200, 1000]], [[550, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[14] = [[1200, 180, 3, 5, 2, 1, 0], [3, 3, 0, 0], [[[350, 450], [350, 750], [1050, 1050]], [[3, 5, 7], [0, 60, 0], [10, 10, 10], [3, 5, 10], [75, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 200], [1100, 200], [900, 400], [1100, 400], [1000, 600]], [[600, 200], [300, 1100]], [[600, 1000]], []], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[15] = [[1200, 180, 3, 2, 1, 1, 1], [3, 2, 0, 0], [[[350, 750], [1050, 150], [1050, 1050]], [[3, 5, 7], [0, 120, 60], [3, 5, 10], [5, 5, 5], [135, 315, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[600, 200], [800, 200]], [[900, 700]], [[200, 1000]], [[550, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[16] = [[1200, 180, 3, 4, 2, 2, 0], [4, 4, 0, 0], [[[350, 450], [350, 750], [1050, 1050]], [[5, 5, 5], [60, 0, 120], [10, 10, 10], [5, 5, 5], [75, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 200], [1100, 200], [900, 600], [1100, 600]], [[600, 200], [300, 1100]], [[1000, 400], [600, 1000]], []], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[17] = [[1200, 180, 3, 2, 1, 1, 1], [2, 3, 0, 0], [[[350, 750], [1050, 150], [1050, 1050]], [[3, 3, 5], [0, 0, 0], [5, 5, 10], [3, 3, 5], [135, 315, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[600, 200], [800, 200]], [[900, 700]], [[200, 1000]], [[550, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[18] = [[1200, 180, 3, 4, 2, 2, 0], [2, 2, 0, 0], [[[350, 450], [350, 750], [1050, 1050]], [[5, 5, 5], [60, 0, 120], [10, 10, 10], [5, 5, 5], [75, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 200], [1100, 200], [900, 600], [1100, 600]], [[600, 200], [300, 1100]], [[1000, 400], [600, 1000]], []], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[19] = [[1200, 180, 3, 2, 1, 1, 1], [2, 1, 0, 0], [[[350, 750], [1050, 150], [1050, 1050]], [[3, 3, 3], [0, 0, 0], [5, 5, 5], [5, 5, 5], [135, 315, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[600, 200], [800, 200]], [[900, 700]], [[200, 1000]], [[550, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[20] = [[1200, 180, 4, 4, 1, 2, 2], [4, 4, 0, 0], [[[250, 450], [250, 750], [50, 1150], [1150, 1150]], [[5, 5, 5, 5], [60, 0, 0, 120], [10, 10, 10, 10], [5, 5, 5, 5], [75, 135, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 500], [1100, 500], [700, 1100], [900, 1100]], [[600, 200]], [[1000, 700], [600, 700]], [[950, 200], [450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[21] = [[1200, 180, 4, 4, 1, 2, 2], [3, 3, 0, 0], [[[250, 450], [250, 750], [50, 1150], [1150, 1150]], [[5, 5, 5, 5], [0, 0, 0, 120], [5, 5, 10, 10], [5, 5, 5, 5], [75, 135, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 500], [1100, 500], [700, 1100], [900, 1100]], [[600, 200]], [[1000, 700], [600, 700]], [[950, 200], [450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[22] = [[1200, 180, 4, 4, 1, 2, 2], [3, 2, 0, 0], [[[250, 450], [250, 750], [50, 1150], [1150, 1150]], [[3, 3, 5, 7], [60, 0, 0, 120], [3, 5, 3, 10], [5, 5, 5, 5], [75, 135, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 500], [1100, 500], [700, 1100], [900, 1100]], [[600, 200]], [[1000, 700], [600, 700]], [[950, 200], [450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[23] = [[1200, 180, 4, 4, 1, 2, 2], [2, 2, 0, 0], [[[250, 450], [250, 750], [50, 1150], [1150, 1150]], [[3, 5, 7, 8], [0, 0, 0, 120], [8, 8, 8, 8], [3, 3, 3, 3], [75, 135, 135, 225], 1.25, 0.1, 90, 80, 1], [[100, 600], 1], [[900, 500], [1100, 500], [700, 1100], [900, 1100]], [[600, 200]], [[1000, 700], [600, 700]], [[950, 200], [450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]
        self.scenario_para[24] = [[1200, 180, 1, 1, 1, 1, 1], [2, 3, 0, 0], [[[350, 350]], [5, 0, 10, 5, 45, 1.25, 0.1, 90, 80, 0], [[100, 600], 1], [[800, 300]], [[900, 800]], [[1000, 600]], [[450, 1000]]], [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]]