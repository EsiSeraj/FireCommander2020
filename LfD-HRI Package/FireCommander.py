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
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pygame

from pygame.locals import *
from sys import exit
import pickle
import numpy as np
from Dependencies.Utilities import Utilities as Util
import matplotlib.pyplot as plt
import os, sys
import shutil
from Dependencies.WildFireModel import WildFire
from Dependencies.Utilities import HeteroFireBots_Reconn_Env_Utilities
from Dependencies.DemoVisualization import Animation_Reconstruction_Reconn_Utilities
from Dependencies.ScenarioModeParams import scenario_setting

scenario = scenario_setting()

# Package the parameter setting controllers (Text display and text editor) in General Setting Page as a single function
def parameter_Input_Ctl(widget, font, left_Upper_Pos, ctl_Text, index):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_index = QLabel(widget)
    text_index.setGeometry(left_upper_x, left_upper_y, 20, 30)
    text_index.setFont(font)
    text_index.setObjectName(edit_Title + 'Index')
    text_index.setText(str(index) + '. ')

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x + 20, left_upper_y, 300, 30)
    text_Display.setFont(font)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_Edit = QLineEdit(widget)
    text_Edit.setGeometry(left_upper_x + 320, left_upper_y, 80, 30)
    text_Edit.setFont(font)
    text_Edit.setObjectName(edit_Title + 'Edit')
    setattr(widget, edit_Title + '_Edit', text_Edit)

# Package the parameter display controllers (Text display and text editor) in Fire Setting Page as a single function
def parameter_Display_Ctl_Extended(widget, font, font_Bold, left_Upper_Pos, ctl_Text, para_value):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x, left_upper_y, 430, 30)
    text_Display.setFont(font_Bold)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_para_Display = QLabel(widget)
    text_para_Display.setGeometry(left_upper_x + 450, left_upper_y, 80, 30)
    text_para_Display.setFont(font)
    text_para_Display.setObjectName(edit_Title + 'Display')
    text_para_Display.setText(para_value)
    setattr(widget, edit_Title + '_Display', text_para_Display)

# Package the parameter display controllers (Text and digit display) in the Target Setting Page as a single function
def parameter_Display_Ctl(widget, font, font_Bold, left_Upper_Pos, ctl_Text, para_value):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x, left_upper_y, 280, 30)
    text_Display.setFont(font_Bold)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_para_Display = QLabel(widget)
    text_para_Display.setGeometry(left_upper_x + 300, left_upper_y, 80, 30)
    text_para_Display.setFont(font)
    text_para_Display.setObjectName(edit_Title + 'Display')
    text_para_Display.setText(para_value)
    setattr(widget, edit_Title + '_Display', text_para_Display)

# Package the parameter display controllers (Coordinates input) in the Target Setting Page as a single function
def environment_Setting_Ctl(widget, font, left_Upper_Pos, ctl_Text, index):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x, left_upper_y, 160, 30)
    text_Display.setFont(font)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_Edit = QLineEdit(widget)
    text_Edit.setGeometry(left_upper_x + 200, left_upper_y, 180, 30)
    text_Edit.setFont(font)
    text_Edit.setObjectName(edit_Title + 'Edit')
    text_Edit.setInputMask('A-99;_')
    text_Edit.textChanged.connect(lambda: text_Edit_function(widget, text_Edit.text(), index))
    setattr(widget, edit_Title + '_Edit', text_Edit)

    def text_Edit_function(widget, str, index):
        widget.raw_list[index] = str
        widget.applied_flag = 0
        font_pe = QPalette()
        widget.applied_display.setGeometry(240, 650, 160, 40)
        font_pe.setColor(QPalette.WindowText, Qt.red)
        widget.applied_display.setText('Not Applied')
        widget.applied_display.setPalette(font_pe)

# Package the parameter display controllers (Text display and single digit editor) in Fire Uniform Setting Page as a single function
def environment_Setting_Ctl_normal(widget, font, left_Upper_Pos, ctl_Text, index):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_index = QLabel(widget)
    text_index.setGeometry(left_upper_x, left_upper_y, 30, 30)
    text_index.setFont(font)
    text_index.setObjectName(edit_Title + 'Index')
    text_index.setText(str(index) + '. ')

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x + 30, left_upper_y, 420, 30)
    text_Display.setFont(font)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_Edit = QLineEdit(widget)
    text_Edit.setGeometry(left_upper_x + 450, left_upper_y, 860, 30)
    text_Edit.setFont(font)
    text_Edit.setObjectName(edit_Title + 'Edit')
    setattr(widget, edit_Title + '_Edit', text_Edit)

# Package the parameter display controllers (Text display and float digit editor) for fire score computation
# in Fire Setting Page as a single function
def environment_Setting_Ctl_Float(widget, font, left_Upper_Pos, ctl_Text, index):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_index = QLabel(widget)
    text_index.setGeometry(left_upper_x, left_upper_y, 30, 30)
    text_index.setFont(font)
    text_index.setObjectName(edit_Title + 'Index')
    text_index.setText(str(index) + '. ')

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x + 30, left_upper_y, 420, 30)
    text_Display.setFont(font)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_Edit1 = QLineEdit(widget)
    text_Edit1.setGeometry(left_upper_x + 450, left_upper_y, 40, 30)
    text_Edit1.setFont(font)
    text_Edit1.setObjectName(edit_Title + 'Edit1')
    setattr(widget, edit_Title + '_Edit1', text_Edit1)

    text_dot = QLabel(widget)
    text_dot.setGeometry(left_upper_x + 490, left_upper_y, 20, 30)
    text_dot.setFont(font)
    text_dot.setObjectName(edit_Title + 'Dot')
    text_dot.setText('. ')
    setattr(widget, edit_Title + '_Dot1', text_dot)

    text_Edit2 = QLineEdit(widget)
    text_Edit2.setGeometry(left_upper_x + 510, left_upper_y, 800, 30)
    text_Edit2.setFont(font)
    text_Edit2.setObjectName(edit_Title + 'Edit2')
    setattr(widget, edit_Title + '_Edit2', text_Edit2)

# Package the parameter display controllers (Text display for multiple digit editor) in Fire Specific Setting Page as a single function
def environment_Setting_Ctl_multi(widget, font, left_Upper_Pos, ctl_Text, index, number):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_index = QLabel(widget)
    text_index.setGeometry(left_upper_x, left_upper_y, 30, 30)
    text_index.setFont(font)
    text_index.setObjectName(edit_Title + 'Index')
    text_index.setText(str(index) + '. ')

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x + 30, left_upper_y, 400, 30)
    text_Display.setFont(font)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

# Package the parameter display controllers (Multiple digit editor) in Fire Specific Setting Page as a single function
def environment_Setting_Ctl_multi_input(widget, font, left_Upper_Pos, ctl_Text, index, i):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_Edit = QLineEdit(widget)
    text_Edit.setGeometry(left_upper_x + 450 + 160 * i, left_upper_y, 140, 30)
    text_Edit.setFont(font)
    text_Edit.setObjectName(edit_Title + 'Edit')
    text_Edit.setValidator(QIntValidator())
    text_Edit.textChanged.connect(lambda: text_Edit_function(widget, text_Edit.text(), index, i))
    setattr(widget, edit_Title + '_Edit' + str(i + 1), text_Edit)

    def text_Edit_function(widget, raw_str, index, i):
        if len(raw_str) == 0:
            widget.raw_list[index - 1][i] = 0
        else:
            widget.raw_list[index - 1][i] = int(str(raw_str))

# Package the parameter display controllers (Multiple coordinates display) in Fire Specific Setting Page as a single function
def fire_coord_display_multi(widget, font, left_Upper_Pos, ctl_Text, i, coord):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text
    x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    raw_list = [round((coord[0] + 50) // 100) - 1, round((coord[1] + 50) // 100)]
    out_str = str(x_coord_list[raw_list[0]]) + '-' + str(raw_list[1] // 10) + str(raw_list[1] % 10)

    text_Edit = QLabel(widget)
    text_Edit.setGeometry(left_upper_x + 450 + 160 * i, left_upper_y, 140, 30)
    text_Edit.setFont(font)
    text_Edit.setObjectName(edit_Title + 'Display')
    text_Edit.setText(out_str)

    setattr(widget, edit_Title + '_Display' + str(i + 1), text_Edit)

# Package the parameter display controllers (Text display and Text editor) as a single function
def parameter_Display_Msg_Extended(widget, font, font_Bold, left_Upper_Pos, ctl_Text, para_value, color):
    (left_upper_x, left_upper_y) = left_Upper_Pos
    (text_Title, edit_Title) = ctl_Text

    text_Display = QLabel(widget)
    text_Display.setGeometry(left_upper_x, left_upper_y, 600, 30)
    text_Display.setFont(font_Bold)
    text_Display.setObjectName(edit_Title)
    text_Display.setText(text_Title)
    setattr(widget, edit_Title, text_Display)

    text_para_Display = QLabel(widget)
    text_para_Display.setGeometry(left_upper_x + 620, left_upper_y, 300, 30)
    text_para_Display.setFont(font)
    text_para_Display.setObjectName(edit_Title + 'Display')
    text_para_Display.setText(para_value)
    setattr(widget, edit_Title + '_Display', text_para_Display)

    font_pe = QPalette()
    font_pe.setColor(QPalette.WindowText, color)
    text_para_Display.setPalette(font_pe)

# Separate the input list in the heterogeneous agent setting page
def list_division(list, num):
    out_list = []
    for i in range(num):
        out_list.append([int(list[2 * i]), int(list[2 * i + 1])])
    return out_list

# Separate the input list in the homogeneous agent setting page
def list_division_single(list, num):
    out_list = []
    for i in range(num):
        out_list.append([int(list[i])])
    return out_list

# Determine whether the given value fits the upper and lower bound
def digit_validate(list, lower_bound, upper_bound, length):
    determine_flag = True
    for i in range(length):
        if int(list[i]) < lower_bound or int(list[i]) > upper_bound:
            determine_flag = False
            break
    return determine_flag

# Determine whether the given value fits the upper and lower bound for the flight height list
def digit_validate_double(list, lower_bound, mid_bound, upper_bound, length):
    determine_flag = 0
    if mid_bound > 0:
        for i in range(length):
            if int(list[2 * i]) < lower_bound or int(list[2 * i]) > mid_bound:
                determine_flag = 1
                break
            elif int(list[2 * i + 1]) < mid_bound or int(list[2 * i + 1]) > upper_bound:
                determine_flag = 2
                break
        return determine_flag
    else:
        for i in range(length):
            if int(list[2 * i]) < lower_bound or int(list[2 * i]) > min(int(list[2 * i + 1]), upper_bound):
                determine_flag = 1
                break
            elif int(list[2 * i + 1]) < max(int(list[2 * i]), upper_bound) or int(list[2 * i + 1]) > upper_bound:
                determine_flag = 2
                break
        return determine_flag

simulated_flag = 0

username = "Tim"

# Initialize the directory structure
class file_initialize():
    def __init__(self, scenario_num):
        super(file_initialize, self).__init__()
        self.scenario_num = scenario_num
        self.init_ui()

    def init_ui(self):
        global username
        for i in range(self.scenario_num):
            if not os.path.exists('Dependencies/Scenario_Data/Scenario#' + str(i+1) + "/" + username):
                os.makedirs('Dependencies/Scenario_Data/Scenario#' + str(i+1) + "/" + username)
            if not os.path.exists('Dependencies/Scenario_Data/Scenario#' + str(i+1) + "/" + username + '/Raw_Images'):
                    os.makedirs('Dependencies/Scenario_Data/Scenario#' + str(i+1) + "/" + username + '/Raw_Images')
        if not os.path.exists('Dependencies/Open_World_Data/' + username):
                os.makedirs('Dependencies/Open_World_Data/' + username)
        if not os.path.exists('Dependencies/Open_World_Data/' + username + '/Raw_Images'):
                os.makedirs('Dependencies/Open_World_Data/' + username + '/Raw_Images')

# The welcome page
class welcome(QMainWindow):
    def __init__(self):
        super(welcome, self).__init__()
        self.init_ui()

    def init_ui(self):
        global simulated_flag
        simulated_flag = 0

        font_button = QFont('arial')
        font_button.setPointSize(28)

        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background_Screen1.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Welcome')

        self.open_world_mode = QPushButton(self)
        self.open_world_mode.setGeometry(250, 650, 300, 150)
        self.open_world_mode.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_150.png)}")
        self.open_world_mode.setFont(font_button)
        self.open_world_mode.setText("Open-world \n Mode")
        self.open_world_mode.clicked.connect(self.open_world_mode_function)

        self.version = QPushButton(self)
        self.version.setGeometry(650, 650, 300, 75)
        self.version.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.version.setFont(font_button)
        self.version.setText("Version 1.1")
        self.version.clicked.connect(self.version_function)

        self.tutorial = QPushButton(self)
        self.tutorial.setGeometry(650, 750, 300, 75)
        self.tutorial.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.tutorial.setFont(font_button)
        self.tutorial.setText("TUTORIAL")
        self.tutorial.clicked.connect(self.tutorial_function)

        self.scenario_mode = QPushButton(self)
        self.scenario_mode.setGeometry(1050, 650, 300, 150)
        self.scenario_mode.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_150.png)}")
        self.scenario_mode.setFont(font_button)
        self.scenario_mode.setText("Scenario Mode")
        self.scenario_mode.clicked.connect(self.scenario_function)

    def open_world_mode_function(self):
        self.hide()
        self.screen = open_world_mode([1200, 180, 0, 0, 0, 0, 0], [0, 0, 0, 0])
        self.screen.show()

    def scenario_function(self):
        self.hide()
        self.screen = scenario_mode()
        self.screen.show()

    def tutorial_function(self):
        self.hide()
        self.screen = tutorial([], [], [], [], -1)
        self.screen.show()

    def version_function(self):
        self.show()
# Load the scenario parameters
scenario = scenario_setting()

# The scenario mode page
class scenario_mode(QWidget):
    def __init__(self):
        super(scenario_mode, self).__init__()
        self.num_scenarios = 24
        self.row = 0
        self.col = 0
        self.idx_selected = -1
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Scenario Mode')

        font = QFont('arial')
        font.setPointSize(28)

        font_button = QFont('arial')
        font_button.setPointSize(26)

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(140, 110, 300, 40)
        self.setting_init.setFont(font)
        self.setting_init.setObjectName("Easy")
        self.setting_init.setText("Easy Scenarios")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(645, 110, 330, 40)
        self.setting_init.setFont(font)
        self.setting_init.setObjectName("Moderate")
        self.setting_init.setText("Moderate Scenarios")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(1225, 110, 300, 40)
        self.setting_init.setFont(font)
        self.setting_init.setObjectName("Hard")
        self.setting_init.setText("Hard Scenarios")

        self.back = QPushButton(self)
        self.back.setGeometry(350, 810, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.practice = QPushButton(self)
        self.practice.setGeometry(950, 810, 300, 75)
        self.practice.setFont(font_button)
        self.practice.setText('Practice')
        self.practice.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.practice.clicked.connect(self.practice_mode)

        for i in range(self.num_scenarios):
            self.col = i % 6
            self.row = i // 6
            self.scenario_button = QPushButton(self)
            self.scenario_button.setGeometry(120 + self.row * 360, 180 + self.col * 100, 300, 75)
            self.scenario_button.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
            self.scenario_button.setFont(font_button)
            self.scenario_button.setObjectName("Scenario_" + str(i + 1))
            self.scenario_button.clicked.connect(lambda:self.scenario_button_function(self.sender().text()))
            self.scenario_button.setText("Scenario #" + str(i + 1))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(210,255,210)))
        painter.drawRect(100, 90, 340, 700)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(178,222,252)))
        painter2.drawRect(460, 90, 700, 700)

        painter3 = QPainter(self)
        painter3.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter3.setBrush(QBrush(QColor(255,210,210)))
        painter3.drawRect(1180, 90, 340, 700)

    def scenario_button_function(self, text):
        if len(text) == 11:
            i = int(text[10]) - 1
        else:
            i = int(text[10] + text[11]) - 1
        self.hide()
        self.screen = tutorial(scenario.scenario_para[i][0], scenario.scenario_para[i][1], scenario.scenario_para[i][2], scenario.scenario_para[i][3], i + 1)
        self.screen.show()

    def practice_mode(self):
        self.hide()
        self.screen = tutorial(scenario.scenario_para[24][0], scenario.scenario_para[24][1], scenario.scenario_para[24][2], scenario.scenario_para[24][3], -1)
        self.screen.show()

    def back_function(self):
        self.hide()
        self.screen = welcome()
        self.screen.show()

# The tutorial page
class tutorial(QWidget):
    def __init__(self, environment_para, robo_team_para, set_loci, adv_setting_list, scenario_idx):
        super(tutorial, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_loci
        self.adv_setting_list = adv_setting_list
        self.scenario_idx = scenario_idx
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Tutorial')

        font_button = QFont('arial')
        font_button.setPointSize(28)

        tutorial1 = QLabel(self)
        tutorial1.setPixmap(QPixmap('./Dependencies/Images/Tutorial1_3.png'))
        tutorial1.setGeometry(60, 70, 1000, 800)

        tutorial2 = QLabel(self)
        tutorial2.setGeometry(1065, 50, 500, 744)
        if self.scenario_idx == 0:
            tutorial2.setPixmap(QPixmap('./Dependencies/Images/Tutorial2.png'))
        else:
            tutorial2.setPixmap(QPixmap('./Dependencies/Images/Tutorial2_2.png'))

        self.menu = QPushButton(self)
        self.menu.setGeometry(1075, 795, 200, 100)
        self.menu.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_200_100.png)}")
        self.menu.setFont(font_button)
        self.menu.setText('Menu')
        self.menu.clicked.connect(self.menu_function)

        self.start = QPushButton(self)
        self.start.setGeometry(1325, 795, 200, 100)
        self.start.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_200_100.png)}")
        self.start.setFont(font_button)
        self.start.setText('Start')
        self.start.clicked.connect(self.start_function)

    def menu_function(self):
        self.hide()
        self.screen = welcome()
        self.screen.show()

    def start_function(self):
        global simulated_flag
        if (len(self.environment_para) == 0) or (len(self.robo_team_para) == 0) or (len(self.set_loci) == 0) or (len(self.adv_setting_list)==0):
            QMessageBox.warning(self, 'Warning', 'Please finish the environment setting or select a default one first')
        else:
            self.hide()
            self.screen = environment_ctl(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, self.scenario_idx)
            if self.scenario_idx == -1:
                simulated_flag = 0
                self.hide()
                self.screen = scenario_mode()
                self.screen.show()
            else:
                if simulated_flag == 1:
                    self.hide()
                    self.screen = game_over(self.scenario_idx)
                    self.screen.show()

# The open-world mode page
class open_world_mode(QWidget):
    def __init__(self, environment_para, robo_team_para):
        super(open_world_mode, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Open World Mode')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.left_y_interval = 40
        self.text_display_len = 300
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font2 = QFont('arial')
        font2.setPointSize(18)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 18, 75)

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("environment_setup")
        self.environment_setup.setText("Environment Setup:")

        self.world_size_index = QLabel(self)
        self.world_size_index.setGeometry(self.left_upper_x, self.left_upper_y + self.left_y_interval, 20, 30)
        self.world_size_index.setFont(font)
        self.world_size_index.setObjectName('world_size_index')
        self.world_size_index.setText('1. ')

        self.world_size = QLabel(self)
        self.world_size.setGeometry(self.left_upper_x + 20, self.left_upper_y + self.left_y_interval, 100, 30)
        self.world_size.setFont(font)
        self.world_size.setObjectName('World Size')
        self.world_size.setText('World Size: ')

        self.world_size_800 = QRadioButton('800',self)
        self.world_size_800.setGeometry(self.left_upper_x + 140, self.left_upper_y + self.left_y_interval, 70, 30)
        self.world_size_800.setFont(font)
        self.world_size_800.setObjectName("World_Size_800")

        self.world_size_1000 = QRadioButton('1000',self)
        self.world_size_1000.setGeometry(self.left_upper_x + 230, self.left_upper_y + self.left_y_interval, 70, 30)
        self.world_size_1000.setFont(font)
        self.world_size_1000.setObjectName("World_Size_1000")

        self.world_size_1200 = QRadioButton('1200',self)
        self.world_size_1200.setGeometry(self.left_upper_x + 320, self.left_upper_y + self.left_y_interval, 70, 30)
        self.world_size_1200.setFont(font)
        self.world_size_1200.setObjectName("World_Size_1200")

        self.world_size_group = QButtonGroup(self)
        self.world_size_group.addButton(self.world_size_800, 11)
        self.world_size_group.addButton(self.world_size_1000, 12)
        self.world_size_group.addButton(self.world_size_1200, 13)
        self.world_size_group.buttonClicked.connect(self.world_size_group_clicked)

        if self.environment_para[0] == 1200:
            self.world_size_800.setChecked(False)
            self.world_size_1000.setChecked(False)
            self.world_size_1200.setChecked(True)
        elif self.environment_para[0] == 1000:
            self.world_size_800.setChecked(False)
            self.world_size_1000.setChecked(True)
            self.world_size_1200.setChecked(False)
        else:
            self.world_size_800.setChecked(True)
            self.world_size_1000.setChecked(False)
            self.world_size_1200.setChecked(False)

        self.duration_index = QLabel(self)
        self.duration_index.setGeometry(self.left_upper_x, self.left_upper_y + 2 * self.left_y_interval, 20, 30)
        self.duration_index.setFont(font)
        self.duration_index.setObjectName('duration_index')
        self.duration_index.setText('2. ')

        self.duration = QLabel(self)
        self.duration.setGeometry(self.left_upper_x + 20, self.left_upper_y + 2 * self.left_y_interval, 100, 30)
        self.duration.setFont(font)
        self.duration.setObjectName('Duration')
        self.duration.setText('Duration: ')

        self.duration_60 = QRadioButton('60',self)
        self.duration_60.setGeometry(self.left_upper_x + 140, self.left_upper_y + 2 * self.left_y_interval, 70, 30)
        self.duration_60.setFont(font)
        self.duration_60.setObjectName("Duration_60")

        self.duration_120 = QRadioButton('120',self)
        self.duration_120.setGeometry(self.left_upper_x + 230, self.left_upper_y + 2 * self.left_y_interval, 70, 30)
        self.duration_120.setFont(font)
        self.duration_120.setObjectName("Duration_120")

        self.duration_180 = QRadioButton('180',self)
        self.duration_180.setGeometry(self.left_upper_x + 320, self.left_upper_y + 2 * self.left_y_interval, 70, 30)
        self.duration_180.setFont(font)
        self.duration_180.setObjectName("Duration_180")

        self.duration_group = QButtonGroup(self)
        self.duration_group.addButton(self.duration_60, 11)
        self.duration_group.addButton(self.duration_120, 12)
        self.duration_group.addButton(self.duration_180, 13)
        self.duration_group.buttonClicked.connect(self.duration_group_clicked)

        if self.environment_para[1] == 180:
            self.duration_60.setChecked(False)
            self.duration_120.setChecked(False)
            self.duration_180.setChecked(True)
        elif self.environment_para[1] == 120:
            self.duration_60.setChecked(False)
            self.duration_120.setChecked(True)
            self.duration_180.setChecked(False)
        else:
            self.duration_60.setChecked(True)
            self.duration_120.setChecked(False)
            self.duration_180.setChecked(False)

        # Fire area number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 3 * self.left_y_interval), ("Number of Fire Areas:", "num_fire_area"), 3)
        self.num_fire_area_Edit.textChanged.connect(lambda:self.num_fire_area_function((self.num_fire_area_Edit.text())))
        self.num_fire_area_Edit.setValidator(QIntValidator())
        self.num_fire_area_Edit.setText(str(self.environment_para[2]))

        # House number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 4 * self.left_y_interval), ("Number of Houses:", "num_houses"), 4)
        self.num_houses_Edit.textChanged.connect(lambda:self.num_houses_function((self.num_houses_Edit.text())))
        self.num_houses_Edit.setValidator(QIntValidator())
        self.num_houses_Edit.setText(str(self.environment_para[3]))

        # Hospital number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 5 * self.left_y_interval), ("Number of Hospitals:", "num_hospitals"), 5)
        self.num_hospitals_Edit.textChanged.connect(lambda:self.num_hospitals_function((self.num_hospitals_Edit.text())))
        self.num_hospitals_Edit.setValidator(QIntValidator())
        self.num_hospitals_Edit.setText(str(self.environment_para[4]))

        # Power station number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 6 * self.left_y_interval), ("Number of Power Station:", "num_Power_Station"), 6)
        self.num_Power_Station_Edit.textChanged.connect(lambda:self.num_Power_Station_function((self.num_Power_Station_Edit.text())))
        self.num_Power_Station_Edit.setValidator(QIntValidator())
        self.num_Power_Station_Edit.setText(str(self.environment_para[5]))

        # Lakes number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 7 * self.left_y_interval), ("Number of Lakes:", "num_Lakes"), 7)
        self.num_Lakes_Edit.textChanged.connect(lambda:self.num_Lakes_function((self.num_Lakes_Edit.text())))
        self.num_Lakes_Edit.setValidator(QIntValidator())
        self.num_Lakes_Edit.setText(str(self.environment_para[6]))

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.left_upper_x, self.left_upper_y + 9 * self.left_y_interval, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("robot_team_setup")
        self.environment_setup.setText("Robot Team Setup:")

        # Perception Agent number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 10 * self.left_y_interval), ("Number of Perception Agents:", "num_Perception_Agent"), 1)
        self.num_Perception_Agent_Edit.textChanged.connect(lambda:self.num_Perception_Agent_function((self.num_Perception_Agent_Edit.text())))
        self.num_Perception_Agent_Edit.setValidator(QIntValidator())
        self.num_Perception_Agent_Edit.setText(str(self.robo_team_para[0]))

        # Action agent number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 11 * self.left_y_interval), ("Number of Action Agents:", "num_Action_Agent"), 2)
        self.num_Action_Agent_Edit.textChanged.connect(lambda:self.num_Action_Agent_function((self.num_Action_Agent_Edit.text())))
        self.num_Action_Agent_Edit.setValidator(QIntValidator())
        self.num_Action_Agent_Edit.setText(str(self.robo_team_para[1]))

        # Hybrid agent number input controller
        parameter_Input_Ctl(self, font, (self.left_upper_x, self.left_upper_y + 12 * self.left_y_interval), ("Number of Hybrid Agents:", "num_Hybrid_Agent"), 3)
        self.num_Hybrid_Agent_Edit.textChanged.connect(lambda:self.num_Hybrid_Agent_function((self.num_Hybrid_Agent_Edit.text())))
        self.num_Hybrid_Agent_Edit.setValidator(QIntValidator())
        self.num_Hybrid_Agent_Edit.setText(str(self.robo_team_para[2]))

        # Tag for homogeneous team input controller
        self.team_mode = QLabel(self)
        self.team_mode.setGeometry(self.left_upper_x, self.left_upper_y + 13 * self.left_y_interval, 300, 30)
        self.team_mode.setFont(font)
        self.team_mode.setObjectName("team_mode")
        self.team_mode.setText("4. Team Mode:")

        self.homo_team = QRadioButton('Homogenous: Agents have the same setting', self)
        self.homo_team.setGeometry(self.left_upper_x, self.left_upper_y + 14 * self.left_y_interval, 400, 30)
        self.homo_team.setFont(font)
        self.homo_team.setObjectName("homo_team")

        self.hetero_team = QRadioButton('Heterogenous: Agents have different settings', self)
        self.hetero_team.setGeometry(self.left_upper_x, self.left_upper_y + 15 * self.left_y_interval, 400, 30)
        self.hetero_team.setFont(font)
        self.hetero_team.setObjectName("hetero_team")

        self.team_mode_group = QButtonGroup(self)
        self.team_mode_group.addButton(self.homo_team, 11)
        self.team_mode_group.addButton(self.hetero_team, 12)
        self.team_mode_group.buttonClicked.connect(self.team_mode_group_clicked)

        if self.robo_team_para[3] == 1:
            self.homo_team.setChecked(False)
            self.hetero_team.setChecked(True)
        else:
            self.homo_team.setChecked(True)
            self.hetero_team.setChecked(False)

        self.back = QPushButton(self)
        self.back.setGeometry(170, 770, 300, 75)
        self.back.setFont(font_button_small)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.reset = QPushButton(self)
        self.reset.setGeometry(700, 770, 300, 75)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1100, 770, 300, 75)
        self.next.setFont(font_button_small)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

        self.instruction = QLabel(self)
        self.instruction.setGeometry(620, 120, 920, 40)
        self.instruction.setFont(font_Title)
        self.instruction.setObjectName("instruction")
        self.instruction.setText("Instruction: ")

        self.instruction1_index = QLabel(self)
        self.instruction1_index.setGeometry(620, 160, 40, 40)
        self.instruction1_index.setFont(font_Bold)
        self.instruction1_index.setObjectName('Index_1')
        self.instruction1_index.setText('  1. ')

        self.instruction1 = QLabel(self)
        self.instruction1.setGeometry(620, 160, 920, 70)
        self.instruction1.setFont(font2)
        self.instruction1.setObjectName("instruction1")
        self.instruction1.setText("      The environment setup and robot team setup on the left of the screen \n \r" +
        "define the number of each object group. All the inputs are required.")

        self.instruction2_index = QLabel(self)
        self.instruction2_index.setGeometry(620, 235, 40, 30)
        self.instruction2_index.setFont(font_Bold)
        self.instruction2_index.setObjectName('Index_2')
        self.instruction2_index.setText('  2. ')

        self.instruction2 = QLabel(self)
        self.instruction2.setGeometry(620, 240, 920, 105)
        self.instruction2.setFont(font2)
        self.instruction2.setObjectName("instruction2")
        self.instruction2.setText("      The set location pages specify the location of each object. Each pages \n \r" +
        "contains the location setting for a specific object. The user must press 'Apply'\n \r" +
        "first to view the approximate position then the 'Next >>' is allowed. All the \n \r" +
        "inputs are required.")

        self.instruction3_index = QLabel(self)
        self.instruction3_index.setGeometry(620, 362, 40, 40)
        self.instruction3_index.setFont(font_Bold)
        self.instruction3_index.setObjectName('Index_3')
        self.instruction3_index.setText('  3. ')

        self.instruction3 = QLabel(self)
        self.instruction3.setGeometry(620, 355, 920, 140)
        self.instruction3.setFont(font2)
        self.instruction3.setObjectName("instruction3")
        self.instruction3.setText("      The advanced setting specifies the information of the robot team. The \n \r" +
        "choice is specified in the robot team setup section, though the choice could \n \r" +
        "be changed through the button below. This section is an optional one, while \n \r" +
        "the default setting is the homogeneous value.")

        self.instruction4_index = QLabel(self)
        self.instruction4_index.setGeometry(620, 495, 40, 40)
        self.instruction4_index.setFont(font_Bold)
        self.instruction4_index.setObjectName('Index_4')
        self.instruction4_index.setText('  4. ')

        self.instruction4 = QLabel(self)
        self.instruction4.setGeometry(620, 495, 920, 70)
        self.instruction4.setFont(font2)
        self.instruction4.setObjectName("instruction4")
        self.instruction4.setText("      The homogeneous setting assumes all the robots share the same setting.\n \r" +
        "Only one input is required for all robots.")

        self.instruction5_index = QLabel(self)
        self.instruction5_index.setGeometry(620, 574, 40, 40)
        self.instruction5_index.setFont(font_Bold)
        self.instruction5_index.setObjectName('Index_5')
        self.instruction5_index.setText('  5. ')

        self.instruction5 = QLabel(self)
        self.instruction5.setGeometry(620, 580, 920, 140)
        self.instruction5.setFont(font2)
        self.instruction5.setObjectName("instruction5")
        self.instruction5.setText("      The heterogenous setting assumes all the robots have different settings.\n \r" +
        "A specific input value is required for each robot in the teams. An error will be \n \r" +
        "sent if the input length does not match the robot number mention in the robot\n \r" +
        "setup section. If the setting is not specified, a default value will be assigned.\n \r")

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 340)
        painter.drawRect(110, 450, 420, 300)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(600, 110, 870, 640)

    def world_size_group_clicked(self):
        if self.world_size_group.checkedId() == 11:
            self.environment_para[0] = 800
        elif self.world_size_group.checkedId() == 12:
            self.environment_para[0] = 1000
        else:
            self.environment_para[0] = 1200

    def duration_group_clicked(self):
        if self.duration_group.checkedId() == 11:
            self.environment_para[1] = 60
        elif self.duration_group.checkedId() == 12:
            self.environment_para[1] = 120
        else:
            self.environment_para[1] = 180

    def num_fire_area_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.environment_para[2] = 0
        else:
            self.environment_para[2] = int(str(edit_str))

    def num_houses_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.environment_para[3] = 0
        else:
            self.environment_para[3] = int(str(edit_str))

    def num_hospitals_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.environment_para[4] = 0
        else:
            self.environment_para[4] = int(str(edit_str))

    def num_Power_Station_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.environment_para[5] = 0
        else:
            self.environment_para[5] = int(str(edit_str))

    def num_Lakes_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.environment_para[6] = 0
        else:
            self.environment_para[6] = int(str(edit_str))

    def num_Perception_Agent_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.robo_team_para[0] = 0
        else:
            self.robo_team_para[0] = int(str(edit_str))

    def num_Action_Agent_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.robo_team_para[1] = 0
        else:
            self.robo_team_para[1] = int(str(edit_str))

    def num_Hybrid_Agent_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.robo_team_para[2] = 0
        else:
            self.robo_team_para[2] = int(str(edit_str))

    def team_mode_group_clicked(self):
        if self.team_mode_group.checkedId() == 11:
            self.robo_team_para[3] = 0
        else:
            self.robo_team_para[3] = 1

    def back_function(self):
        self.hide()
        self.screen = welcome()
        self.screen.show()

    def reset_function(self):
        self.hide()
        self.screen = open_world_mode([1200, 180, 0, 0, 0, 0, 0], [0, 0, 0, 0])
        self.screen.show()

    def num_fire_determination(self):
        if self.environment_para[2] > 5:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of fire regions could not exceed 5')
            return False
        elif self.environment_para[2] == 0:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of fire regions could not be 0')
            return False
        else:
            return True

    def num_house_determination(self):
        if self.environment_para[3] > 5:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of houses could not exceed 5')
            return False
        else:
            return True

    def num_hospital_determination(self):
        if self.environment_para[4] > 5:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of hospitals could not exceed 5')
            return False
        else:
            return True

    def num_power_determination(self):
        if self.environment_para[5] > 5:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of power stations could not exceed 5')
            return False
        else:
            return True

    def num_lake_determination(self):
        if self.environment_para[6] > 5:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of lakes could not exceed 5')
            return False
        else:
            return True

    def num_agent_determination(self):
        if (self.robo_team_para[0] == 0) or (self.robo_team_para[1] == 0) or (self.robo_team_para[2] == 0):
            if (self.robo_team_para[0] == 0) and (self.robo_team_para[2] == 0):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The number of the perception and hybrid agents could not be 0 at the same time')
                return False
            elif (self.robo_team_para[1] == 0) and (self.robo_team_para[2] == 0):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The number of the action and hybrid agents could not be 0 at the same time')
                return False
            else:
                return True
        else:
            if (self.robo_team_para[0] + self.robo_team_para[1] + self.robo_team_para[2]) > 9:
                QMessageBox.warning(self, 'Warning', 'Invalid input: The total number of agents could not exceed 9')
                return False
            else:
                return True

    def constraint_check(self):
        default = False
        if self.num_fire_determination():
            if self.num_house_determination():
                if self.num_hospital_determination():
                    if self.num_power_determination():
                        if self.num_lake_determination():
                            if self.num_agent_determination():
                                default = True
        return default

    def next_function(self):
        if self.constraint_check():
            applied_flag = [0, 0, 0, 0, 0, 0, 0]
            current_grid = list(np.zeros((12, 12), dtype=float))
            self.hide()
            self.screen = fire_location_define(self.environment_para, self.robo_team_para, [[], [5, 0, 10, 5, 45, 1.25, 0.1, 90, 80, 0], [[], 1], [], [], [], []], current_grid, applied_flag)
            self.screen.show()

# The fire location setting
class fire_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(fire_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[0]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.update_flag = 0
        self.global_applied_flag = applied_flag

        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Fire Region Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = []
        for i in range(self.environment_para[2]):
            self.raw_list.append('')
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("Fire_Region_Setting")
        self.setting_init.setText("Fire Region Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Fire Regions:", "num_fires"), str(self.environment_para[2]))

        self.size_note = QLabel(self)
        self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 60, 30)
        self.size_note.setFont(font_Bold)
        self.size_note.setObjectName('Fire Region Size Note')
        self.size_note.setText('Note: ')

        self.fire_size = QLabel(self)
        self.fire_size.setGeometry(self.left_upper_x + 60, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.fire_size.setFont(font)
        self.fire_size.setObjectName('Fire Region Size')
        self.fire_size.setText('A 1 × 1 Grid will be Marked')

        if self.environment_para[2] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Fire")
            self.location_init.setText("No Fire Region in this Environment")
        else:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len, self.text_height)
            self.location_init.setFont(font_Bold)
            self.location_init.setObjectName("Fire_location")
            self.location_init.setText("Fire Region Locations:")

            for i in range(self.environment_para[2]):
                # Initialize fire center input controller
                environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (4.5 + i) * self.left_y_interval),
                                        ("Fire Region #" + str(i + 1) + ": ", "fire" + str(i + 1) + "_center"), i)
        self.word_plot(font_grid)
        self.set_initial_value_function()

        self.applied_flag = self.global_applied_flag[0]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 1:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(255, 0, 0)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def fire_location_determination(self, index):
        default = False
        if self.environment_para[2] > 0:
            raw_coord_init = self.raw_list[index].split('-')
            if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
            else:
                if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
                else:
                    if (raw_coord_init[0] > self.x_coord_list[self.grid_size]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size) or (int(raw_coord_init[1]) < 1):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                    else:
                        raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                        if not self.overlap_check(raw_coord[0], raw_coord[1] - 1):
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Fire region ' + str(index + 1) + ' overlaps with existing objects')
                        else:
                            default = True
        return default

    def array_to_string(self, list):
        raw_list = [round((list[0] + 50)//100) - 1, round((list[1] + 50)//100)]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.out_list[i])

    def set_initial_value_function(self):
        if self.environment_para[2] > 0:
            self.fire1_center_Edit.setText(self.raw_list[0])
            if self.environment_para[2] > 1:
                self.fire2_center_Edit.setText(self.raw_list[1])
            if self.environment_para[2] > 2:
                self.fire3_center_Edit.setText(self.raw_list[2])
            if self.environment_para[2] > 3:
                self.fire4_center_Edit.setText(self.raw_list[3])
            if self.environment_para[2] > 4:
                self.fire5_center_Edit.setText(self.raw_list[4])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        if self.environment_para[2] > 0:
            self.fire1_center_Edit.clear()
            self.raw_list[0] = ''
            if self.environment_para[2] > 1:
                self.fire2_center_Edit.clear()
                self.raw_list[1] = ''
            if self.environment_para[2] > 2:
                self.fire3_center_Edit.clear()
                self.raw_list[2] = ''
            if self.environment_para[2] > 3:
                self.fire4_center_Edit.clear()
                self.raw_list[3] = ''
            if self.environment_para[2] > 4:
                self.fire5_center_Edit.clear()
                self.raw_list[4] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if (self.current_grid[pos_x][pos_y] == 0) or (self.current_grid[pos_x][pos_y] == 1):
            if (self.temp_grid[pos_x][pos_y] == 0) or (self.temp_grid[pos_x][pos_y] == 1):
                default = True
        return default

    def apply_function(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 1:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 1:
                    self.temp_grid[i][j] = 0

        determine_flag = True
        for i in range(self.environment_para[2]):
            if (self.environment_para[2] > i) and (not self.fire_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                break

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 1

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[0] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 1:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[0] = 0
        self.hide()
        self.screen = open_world_mode(self.environment_para, self.robo_team_para)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each fire region first')
        else:
            determine_flag = True
            for i in range(self.environment_para[2]):
                if (self.environment_para[2] > i) and (not self.fire_location_determination(i)):
                    determine_flag = False
                    self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                    break

            if determine_flag:
                new_list = []
                for i in range(self.environment_para[2]):
                    raw_coord = self.raw_list[i].split('-')
                    new_list.append([(self.x_coord_list.index(raw_coord[0]) + 1) * 100 - 50, int(raw_coord[1]) * 100 - 50])
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 1
                self.set_Loci[0] = new_list
                self.global_applied_flag[0] = self.applied_flag
                self.hide()
                if self.set_Loci[1][9] == 1:
                    self.screen = fire_setting_spec(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
                else:
                    self.screen = fire_setting_uni(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
                self.screen.show()

# The uniform fire parameter setting page
class fire_setting_uni(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(fire_setting_uni, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_Loci
        self.out_list = set_Loci[1]
        self.applied_flag = applied_flag
        self.current_grid = current_grid
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Fire Setting (Uniform)')

        self.right_upper_x = 160
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 500
        self.text_height = 40

        self.raw_list = [0, 0, 0, 0, 0, [0, 0], [0, 0], 0, 0, 0]

        font = QFont('arial')
        font.setPointSize(14)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button2 = QFont('arial')
        font_button2.setPointSize(20)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.right_upper_x, self.right_upper_y, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("fire_setting_uni")
        self.environment_setup.setText("Fire Setting (Uniform):")

        # House number input controller
        parameter_Display_Ctl_Extended(self, font, font_Bold,
                              (self.right_upper_x, self.right_upper_y + 1.5 * self.left_y_interval),
                              ("Number of Fire Regions:", "num_fires"), str(self.environment_para[2]))

        if self.environment_para[2] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Fire")
            self.location_init.setText("No Fire Region in this Environment")
        else:
            self.fire_region = QLabel(self)
            self.fire_region.setGeometry(self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval, 320,
                                               self.text_height)
            self.fire_region.setFont(font_Bold)
            self.fire_region.setObjectName("fire_region")
            self.fire_region.setText("Current Fire Regions:")

            for i in range(self.environment_para[2]):
                fire_coord_display_multi(self, font,  (self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval),
                                    ("fire_region_name", "fire_region_name"), i, self.set_loci[0][i])


            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 3.5 * self.left_y_interval),
                                    ("Number of Fire Fronts in each Region: ", "num_fire_spot"), 1)
            self.num_fire_spot_Edit.textChanged.connect(lambda: self.num_fire_spot_function((self.num_fire_spot_Edit.text())))
            self.num_fire_spot_Edit.setValidator(QIntValidator())
            self.num_fire_spot_Edit.setText(str(self.out_list[0]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 4.5 * self.left_y_interval),
                                    ("Fire Delay Time (Min: 0, Max: 180):", "fire_delay"), 2)
            self.fire_delay_Edit.textChanged.connect(lambda: self.fire_delay_function((self.fire_delay_Edit.text())))
            self.fire_delay_Edit.setValidator(QIntValidator())
            self.fire_delay_Edit.setText(str(self.out_list[1]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 5.5 * self.left_y_interval),
                                    ("Fuel Coefficient (Min: 2, Max: 20):", "fuel_coef"), 3)
            self.fuel_coef_Edit.textChanged.connect(lambda: self.fuel_coef_function((self.fuel_coef_Edit.text())))
            self.fuel_coef_Edit.setValidator(QIntValidator())
            self.fuel_coef_Edit.setText(str(self.out_list[2]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 6.5 * self.left_y_interval),
                                    ("Wind Speed (Min: 2, Max: 10):", "wind_speed"), 4)
            self.wind_speed_Edit.textChanged.connect(lambda: self.wind_speed_function((self.wind_speed_Edit.text())))
            self.wind_speed_Edit.setValidator(QIntValidator())
            self.wind_speed_Edit.setText(str(self.out_list[3]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 7.5 * self.left_y_interval),
                                    ("Wind Direction (0 - 360 Degrees):", "wind_direction"), 5)
            self.wind_direction_Edit.textChanged.connect(lambda: self.wind_direction_function((self.wind_direction_Edit.text())))
            self.wind_direction_Edit.setValidator(QIntValidator())
            self.wind_direction_Edit.setText(str(self.out_list[4]))

            environment_Setting_Ctl_Float(self, font,  (self.right_upper_x, self.right_upper_y + 8.5 * self.left_y_interval),
                                    ("Temporal Penalty Coefficient (Min: 0, Max: 2):", "temporal_penalty_coef"), 6)
            self.temporal_penalty_coef_Edit1.textChanged.connect(lambda: self.temporal_penalty_coef_function1((self.temporal_penalty_coef_Edit1.text())))
            self.temporal_penalty_coef_Edit1.setValidator(QIntValidator())
            self.temporal_penalty_coef_Edit1.setText(str(int(np.floor(self.out_list[5]))))

            self.temporal_penalty_coef_Edit2.textChanged.connect(lambda: self.temporal_penalty_coef_function2((self.temporal_penalty_coef_Edit2.text())))
            self.temporal_penalty_coef_Edit2.setValidator(QIntValidator())
            self.temporal_penalty_coef_Edit2.setText(str(self.out_list[5]).split(".")[1])

            environment_Setting_Ctl_Float(self, font,  (self.right_upper_x, self.right_upper_y + 9.5 * self.left_y_interval),
                                    ("Fire Propagation Weight (Min: 0, Max: 1):", "fire_propagation_weight"), 7)
            self.fire_propagation_weight_Edit1.textChanged.connect(lambda: self.fire_propagation_weight_function1((self.fire_propagation_weight_Edit1.text())))
            self.fire_propagation_weight_Edit1.setValidator(QIntValidator())
            self.fire_propagation_weight_Edit1.setText(str(int(np.floor(self.out_list[6]))))

            self.fire_propagation_weight_Edit2.textChanged.connect(lambda: self.fire_propagation_weight_function2((self.fire_propagation_weight_Edit2.text())))
            self.fire_propagation_weight_Edit2.setValidator(QIntValidator())
            self.fire_propagation_weight_Edit2.setText(str(self.out_list[6]).split(".")[1])

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 10.5 * self.left_y_interval),
                                    ("Action Pruning Confidence Level (10% - 100%):", "action_confidence_level"), 8)
            self.action_confidence_level_Edit.textChanged.connect(lambda: self.action_confidence_level_function((self.action_confidence_level_Edit.text())))
            self.action_confidence_level_Edit.setValidator(QIntValidator())
            self.action_confidence_level_Edit.setText(str(self.out_list[7]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 11.5 * self.left_y_interval),
                                    ("Hybrid Pruning Confidence Level (10% - 100%):", "hybrid_confidence_level"), 9)
            self.hybrid_confidence_level_Edit.textChanged.connect(lambda: self.hybrid_confidence_level_function((self.hybrid_confidence_level_Edit.text())))
            self.hybrid_confidence_level_Edit.setValidator(QIntValidator())
            self.hybrid_confidence_level_Edit.setText(str(self.out_list[8]))

        self.back = QPushButton(self)
        self.back.setGeometry(125, 770, 300, 75)
        self.back.setFont(font_button2)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.transfer = QPushButton(self)
        self.transfer.setGeometry(475, 770, 300, 75)
        self.transfer.setFont(font_button2)
        self.transfer.setText('Transfer to \nSpecific Setting')
        self.transfer.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.transfer.clicked.connect(self.transfer_function)

        self.skip = QPushButton(self)
        self.skip.setGeometry(825, 770, 300, 75)
        self.skip.setFont(font_button2)
        self.skip.setText('Skip')
        self.skip.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.skip.clicked.connect(self.skip_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1174, 770, 300, 75)
        self.next.setFont(font_button2)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def paintEvent(self, e):
        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(110, 90, 1400, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(140, 110, 1340, 640)

    def num_fire_spot_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[0] = 0
        else:
            self.raw_list[0] = int(str(edit_str))

    def fire_delay_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[1] = 0
        else:
            self.raw_list[1] = int(str(edit_str))

    def fuel_coef_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[2] = 0
        else:
            self.raw_list[2] = int(str(edit_str))

    def wind_speed_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[3] = 0
        else:
            self.raw_list[3] = int(str(edit_str))

    def wind_direction_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[4] = 0
        else:
            self.raw_list[4] = int(str(edit_str))

    def temporal_penalty_coef_function1(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[5][0] = 0
        else:
            self.raw_list[5][0] = float(str(edit_str))

    def temporal_penalty_coef_function2(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[5][1] = 0
        else:
            self.raw_list[5][1] = float(str(edit_str)) / (10 ** len(edit_str))

    def fire_propagation_weight_function1(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[6][0] = 0
        else:
            self.raw_list[6][0] = float(str(edit_str))

    def fire_propagation_weight_function2(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[6][1] = 0
        else:
            self.raw_list[6][1] = float(str(edit_str)) / (10 ** len(edit_str))

    def action_confidence_level_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[7] = 0
        else:
            self.raw_list[7] = int(str(edit_str))

    def hybrid_confidence_level_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[8] = 0
        else:
            self.raw_list[8] = int(str(edit_str))

    def fire_spot_determination(self):
        default = False
        if (self.raw_list[0] < 1):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The number of fire spots in each region should be larger than 1')
        elif ((self.environment_para[2] * self.raw_list[0]) > 30):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The total number of fire spots should not exceed 30')
        else:
            default = True
        return default

    def fire_delay_determination(self):
        default = False
        if (self.raw_list[1] < 0) or (self.raw_list[1] > 180):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The fire delay time should be between 0 and 180 seconds')
        elif self.raw_list[1] > self.environment_para[1]:
            QMessageBox.warning(self, 'Warning', 'Invalid input: The fire delay time should not exceed the simulation range')
        else:
            default = True
        return default

    def fuel_coef_determination(self):
        default = False
        if (self.raw_list[2] < 2) or (self.raw_list[2] > 20):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The fuel coefficient should be between 2 and 20')
        else:
            default = True
        return default

    def wind_speed_determination(self):
        default = False
        if (self.raw_list[3] < 2) or (self.raw_list[3] > 10):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The wind speed should be between 2 and 10')
        else:
            default = True
        return default

    def wind_direction_determination(self):
        default = False
        if (self.raw_list[4] < 0) or (self.raw_list[4] > 360):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The wind direction should be between 0 and 360')
        else:
            default = True
        return default

    def temporal_penalty_coef_determination(self):
        raw_num = self.raw_list[5][0] + self.raw_list[5][1]
        default = False
        if (raw_num < 0) or (raw_num > 2):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The temporal penalty coefficient should be between 0 and 2')
        else:
            default = True
        return default

    def fire_propagation_weight_determination(self):
        default = False
        raw_num = self.raw_list[6][0] + self.raw_list[6][1]
        if (raw_num < 0) or (raw_num > 1):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The fire propagation weight should be between 0 and 1')
        else:
            default = True
        return default

    def action_confidence_level_determination(self):
        default = False
        if (self.raw_list[7] < 10) or (self.raw_list[7] > 100):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The action confidence level should be between 10 and 100')
        else:
            default = True
        return default

    def hybrid_confidence_level_determination(self):
        default = False
        if (self.raw_list[8] < 10) or (self.raw_list[8] > 100):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The hybrid confidence level should be between 10 and 100')
        else:
            default = True
        return default

    def condition_check(self):
        default = False
        if self.fire_spot_determination():
            if self.fire_delay_determination():
                if self.fuel_coef_determination():
                    if self.wind_speed_determination():
                        if self.wind_direction_determination():
                            if self.temporal_penalty_coef_determination():
                                if self.fire_propagation_weight_determination():
                                    if self.action_confidence_level_determination():
                                        if self.hybrid_confidence_level_determination():
                                            default = True
        return default

    def transmit_value(self):
        for i in range(len(self.out_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = 0
            else:
                self.raw_list[i] = str(self.out_list[i])

    def back_function(self):
        self.hide()
        self.screen = fire_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def transfer_function(self):
        num_fire_front_list = []
        fire_delay_list = []
        fuel_coef_list = []
        wind_speed_list = []
        wind_direction_list = []

        for i in range(self.environment_para[2]):
            num_fire_front_list.append(5)
            fire_delay_list.append(0)
            fuel_coef_list.append(10)
            wind_speed_list.append(5)
            wind_direction_list.append(45)

        self.set_loci[1] = [num_fire_front_list, fire_delay_list, fuel_coef_list, wind_speed_list, wind_direction_list, 1.25, 0.1, 90, 80, 1]
        self.hide()
        self.screen = fire_setting_spec(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def skip_function(self):
        self.set_loci[1] = [5, 0, 10, 5, 45, 1.25, 0.1, 90, 80, 0]
        self.hide()
        self.screen = agent_base_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def next_function(self):
        if self.condition_check():
            self.set_loci[1] = self.raw_list
            self.set_loci[1][5] = self.raw_list[5][0] + self.raw_list[5][1]
            self.set_loci[1][6] = self.raw_list[6][0] + self.raw_list[6][1]
            self.hide()
            self.screen = agent_base_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
            self.screen.show()

# The specific fire setting page
class fire_setting_spec(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(fire_setting_spec, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_Loci
        self.out_list = set_Loci[1]
        self.applied_flag = applied_flag
        self.current_grid = current_grid
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Fire Setting (Specific)')

        self.right_upper_x = 160
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 500
        self.text_height = 40

        self.raw_list = [[], [], [], [], [], [0, 0], [0, 0], 0, 0, 1]

        font = QFont('arial')
        font.setPointSize(14)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button2 = QFont('arial')
        font_button2.setPointSize(20)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.right_upper_x, self.right_upper_y, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("fire_setting_spec")
        self.environment_setup.setText("Fire Setting (Specific):")

        # House number input controller
        parameter_Display_Ctl_Extended(self, font, font_Bold,
                              (self.right_upper_x, self.right_upper_y + 1.5 * self.left_y_interval),
                              ("Number of Fire Regions:", "num_fires"), str(self.environment_para[2]))

        if self.environment_para[2] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Fire")
            self.location_init.setText("No Fire Region in this Environment")
        else:
            self.fire_region = QLabel(self)
            self.fire_region.setGeometry(self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval, 320,
                                               self.text_height)
            self.fire_region.setFont(font_Bold)
            self.fire_region.setObjectName("fire_region")
            self.fire_region.setText("Current Fire Regions:")

            for i in range(self.environment_para[2]):
                fire_coord_display_multi(self, font,  (self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval),
                                    ("fire_region_name", "fire_region_name"), i, self.set_loci[0][i])

            environment_Setting_Ctl_multi(self, font,  (self.right_upper_x, self.right_upper_y + 3.5 * self.left_y_interval),
                                    ("Number of Fire Fronts in each Region: ", "num_fire_spot"), 1, self.environment_para[2])
            for i in range(self.environment_para[2]):
                environment_Setting_Ctl_multi_input(self, font, (self.right_upper_x, self.right_upper_y + 3.5 * self.left_y_interval),
                                              ("Number of Fire Fronts in each Region: ", "num_fire_spot"), 1, i)

            environment_Setting_Ctl_multi(self, font,  (self.right_upper_x, self.right_upper_y + 4.5 * self.left_y_interval),
                                    ("Fire Delay Time (Min: 0, Max: 180):", "fire_delay"), 2, self.environment_para[2])
            for i in range(self.environment_para[2]):
                environment_Setting_Ctl_multi_input(self, font,  (self.right_upper_x, self.right_upper_y + 4.5 * self.left_y_interval),
                                    ("Fire Delay Time (Min: 0, Max: 180):", "fire_delay"), 2, i)

            environment_Setting_Ctl_multi(self, font,  (self.right_upper_x, self.right_upper_y + 5.5 * self.left_y_interval),
                                    ("Fuel Coefficient (Min: 2, Max: 20):", "fuel_coef"), 3, self.environment_para[2])
            for i in range(self.environment_para[2]):
                environment_Setting_Ctl_multi_input(self, font,  (self.right_upper_x, self.right_upper_y + 5.5 * self.left_y_interval),
                                    ("Fuel Coefficient (Min: 2, Max: 20):", "fuel_coef"), 3, i)

            environment_Setting_Ctl_multi(self, font,  (self.right_upper_x, self.right_upper_y + 6.5 * self.left_y_interval),
                                    ("Wind Speed (Min: 2, Max: 10):", "wind_speed"), 4, self.environment_para[2])
            for i in range(self.environment_para[2]):
                environment_Setting_Ctl_multi_input(self, font,  (self.right_upper_x, self.right_upper_y + 6.5 * self.left_y_interval),
                                    ("Wind Speed (Min: 2, Max: 10):", "wind_speed"), 4, i)

            environment_Setting_Ctl_multi(self, font,  (self.right_upper_x, self.right_upper_y + 7.5 * self.left_y_interval),
                                    ("Wind Direction (0 - 360 Degrees):", "wind_direction"), 5, self.environment_para[2])
            for i in range(self.environment_para[2]):
                environment_Setting_Ctl_multi_input(self, font,  (self.right_upper_x, self.right_upper_y + 7.5 * self.left_y_interval),
                                    ("Wind Direction (0 - 360 Degrees):", "wind_direction"), 5, i)

            environment_Setting_Ctl_Float(self, font,  (self.right_upper_x, self.right_upper_y + 8.5 * self.left_y_interval),
                                    ("Temporal Penalty Coefficient (Min: 0, Max: 2):", "temporal_penalty_coef"), 6)
            self.temporal_penalty_coef_Edit1.textChanged.connect(lambda: self.temporal_penalty_coef_function1((self.temporal_penalty_coef_Edit1.text())))
            self.temporal_penalty_coef_Edit1.setValidator(QIntValidator())
            self.temporal_penalty_coef_Edit1.setText(str(int(np.floor(self.out_list[5]))))

            self.temporal_penalty_coef_Edit2.textChanged.connect(lambda: self.temporal_penalty_coef_function2((self.temporal_penalty_coef_Edit2.text())))
            self.temporal_penalty_coef_Edit2.setValidator(QIntValidator())
            self.temporal_penalty_coef_Edit2.setText(str(self.out_list[5]).split(".")[1])

            environment_Setting_Ctl_Float(self, font,  (self.right_upper_x, self.right_upper_y + 9.5 * self.left_y_interval),
                                    ("Fire Propagation Weight (Min: 0, Max: 1):", "fire_propagation_weight"), 7)
            self.fire_propagation_weight_Edit1.textChanged.connect(lambda: self.fire_propagation_weight_function1((self.fire_propagation_weight_Edit1.text())))
            self.fire_propagation_weight_Edit1.setValidator(QIntValidator())
            self.fire_propagation_weight_Edit1.setText(str(int(np.floor(self.out_list[6]))))

            self.fire_propagation_weight_Edit2.textChanged.connect(lambda: self.fire_propagation_weight_function2((self.fire_propagation_weight_Edit2.text())))
            self.fire_propagation_weight_Edit2.setValidator(QIntValidator())
            self.fire_propagation_weight_Edit2.setText(str(self.out_list[6]).split(".")[1])

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 10.5 * self.left_y_interval),
                                    ("Action Pruning Confidence Level (10% - 100%):", "action_confidence_level"), 8)
            self.action_confidence_level_Edit.textChanged.connect(lambda: self.action_confidence_level_function((self.action_confidence_level_Edit.text())))
            self.action_confidence_level_Edit.setValidator(QIntValidator())
            self.action_confidence_level_Edit.setText(str(self.out_list[7]))

            environment_Setting_Ctl_normal(self, font,  (self.right_upper_x, self.right_upper_y + 11.5 * self.left_y_interval),
                                    ("Hybrid Pruning Confidence Level (10% - 100%):", "hybrid_confidence_level"), 9)
            self.hybrid_confidence_level_Edit.textChanged.connect(lambda: self.hybrid_confidence_level_function((self.hybrid_confidence_level_Edit.text())))
            self.hybrid_confidence_level_Edit.setValidator(QIntValidator())
            self.hybrid_confidence_level_Edit.setText(str(self.out_list[8]))


        self.transmit_value()
        self.set_initial_value_function()

        self.back = QPushButton(self)
        self.back.setGeometry(125, 770, 300, 75)
        self.back.setFont(font_button2)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.transfer = QPushButton(self)
        self.transfer.setGeometry(475, 770, 300, 75)
        self.transfer.setFont(font_button2)
        self.transfer.setText('Transfer to \nUniform Setting')
        self.transfer.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.transfer.clicked.connect(self.transfer_function)

        self.skip = QPushButton(self)
        self.skip.setGeometry(825, 770, 300, 75)
        self.skip.setFont(font_button2)
        self.skip.setText('Skip')
        self.skip.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.skip.clicked.connect(self.skip_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1174, 770, 300, 75)
        self.next.setFont(font_button2)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def paintEvent(self, e):
        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(110, 90, 1400, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(140, 110, 1340, 640)

    def temporal_penalty_coef_function1(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[5][0] = 0
        else:
            self.raw_list[5][0] = float(str(edit_str))

    def temporal_penalty_coef_function2(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[5][1] = 0
        else:
            self.raw_list[5][1] = float(str(edit_str)) / (10 ** len(edit_str))

    def fire_propagation_weight_function1(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[6][0] = 0
        else:
            self.raw_list[6][0] = float(str(edit_str))

    def fire_propagation_weight_function2(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[6][1] = 0
        else:
            self.raw_list[6][1] = float(str(edit_str)) / (10 ** len(edit_str))

    def action_confidence_level_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[7] = 0
        else:
            self.raw_list[7] = int(str(edit_str))

    def hybrid_confidence_level_function(self, edit_str):
        if (len(str(edit_str)) == 0):
            self.raw_list[8] = 0
        else:
            self.raw_list[8] = int(str(edit_str))

    def fire_spot_determination(self):
        default = True
        count = 0
        for i in range(self.environment_para[2]):
            if (self.raw_list[0][i] < 1):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The number of fire spots in each region should be larger than 1')
                default = False
                break
            else:
                count += self.raw_list[0][i]
        if count > 30:
             QMessageBox.warning(self, 'Warning', 'Invalid input: The total number of fire spots should not exceed 30')
             default = False
        return default

    def fire_delay_determination(self):
        default = True
        for i in range(self.environment_para[2]):
            if (self.raw_list[1][i] < 0) or (self.raw_list[1][i] > 180):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The fire delay time should be between 0 and 180 seconds')
                default = False
                break
            elif (self.raw_list[1][i] > self.environment_para[1]):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The fire delay time should not exceed the simulation range')
                default = False
                break
            else:
                default = True
        return default

    def fuel_coef_determination(self):
        default = True
        for i in range(self.environment_para[2]):
            if (self.raw_list[2][i] < 2) or (self.raw_list[2][i] > 20):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The fuel coefficient should be between 2 and 20')
                default = False
                break
        return default

    def wind_speed_determination(self):
        default = True
        for i in range(self.environment_para[2]):
            if (self.raw_list[3][i] < 2) or (self.raw_list[3][i] > 10):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The wind speed should be between 2 and 10')
                default = False
                break
        return default

    def wind_direction_determination(self):
        default = True
        for i in range(self.environment_para[2]):
            if (self.raw_list[4][i] < 0) or (self.raw_list[4][i] > 360):
                QMessageBox.warning(self, 'Warning', 'Invalid input: The wind direction should be between 0 and 360')
                default = False
                break
        return default

    def temporal_penalty_coef_determination(self):
        raw_num = self.raw_list[5][0] + self.raw_list[5][1]
        default = False
        if (raw_num < 0) or (raw_num > 2):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The temporal penalty coefficient should be between 0 and 2')
        else:
            default = True
        return default

    def fire_propagation_weight_determination(self):
        default = False
        raw_num = self.raw_list[6][0] + self.raw_list[6][1]
        if (raw_num < 0) or (raw_num > 1):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The fire propagation weight should be between 0 and 1')
        else:
            default = True
        return default

    def action_confidence_level_determination(self):
        default = False
        if (self.raw_list[7] < 10) or (self.raw_list[7] > 100):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The action confidence level should be between 10 and 100')
        else:
            default = True
        return default

    def hybrid_confidence_level_determination(self):
        default = False
        if (self.raw_list[8] < 10) or (self.raw_list[8] > 100):
            QMessageBox.warning(self, 'Warning', 'Invalid input: The hybrid confidence level should be between 10 and 100')
        else:
            default = True
        return default

    def set_initial_value_function(self):
        if self.environment_para[2] > 0:
            self.num_fire_spot_Edit1.setText(str(self.raw_list[0][0]))
            self.fire_delay_Edit1.setText(str(self.raw_list[1][0]))
            self.fuel_coef_Edit1.setText(str(self.raw_list[2][0]))
            self.wind_speed_Edit1.setText(str(self.raw_list[3][0]))
            self.wind_direction_Edit1.setText(str(self.raw_list[4][0]))
            if self.environment_para[2] > 1:
                self.num_fire_spot_Edit2.setText(str(self.raw_list[0][1]))
                self.fire_delay_Edit2.setText(str(self.raw_list[1][1]))
                self.fuel_coef_Edit2.setText(str(self.raw_list[2][1]))
                self.wind_speed_Edit2.setText(str(self.raw_list[3][1]))
                self.wind_direction_Edit2.setText(str(self.raw_list[4][1]))
            if self.environment_para[2] > 2:
                self.num_fire_spot_Edit3.setText(str(self.raw_list[0][2]))
                self.fire_delay_Edit3.setText(str(self.raw_list[1][2]))
                self.fuel_coef_Edit3.setText(str(self.raw_list[2][2]))
                self.wind_speed_Edit3.setText(str(self.raw_list[3][2]))
                self.wind_direction_Edit3.setText(str(self.raw_list[4][2]))
            if self.environment_para[2] > 3:
                self.num_fire_spot_Edit4.setText(str(self.raw_list[0][3]))
                self.fire_delay_Edit4.setText(str(self.raw_list[1][3]))
                self.fuel_coef_Edit4.setText(str(self.raw_list[2][3]))
                self.wind_speed_Edit4.setText(str(self.raw_list[3][3]))
                self.wind_direction_Edit4.setText(str(self.raw_list[4][3]))
            if self.environment_para[2] > 4:
                self.num_fire_spot_Edit5.setText(str(self.raw_list[0][4]))
                self.fire_delay_Edit5.setText(str(self.raw_list[1][4]))
                self.fuel_coef_Edit5.setText(str(self.raw_list[2][4]))
                self.wind_speed_Edit5.setText(str(self.raw_list[3][4]))
                self.wind_direction_Edit5.setText(str(self.raw_list[4][4]))

    def condition_check(self):
        default = False
        if self.fire_spot_determination():
            if self.fire_delay_determination():
                if self.fuel_coef_determination():
                    if self.wind_speed_determination():
                        if self.wind_direction_determination():
                            if self.temporal_penalty_coef_determination():
                                if self.fire_propagation_weight_determination():
                                    if self.action_confidence_level_determination():
                                        if self.hybrid_confidence_level_determination():
                                            default = True
        return default

    def transmit_value(self):
        for i in range(len(self.out_list) - 5):
            for j in range(len(self.out_list[i])):
                if len(self.out_list) == 0:
                    self.raw_list[i].append('0')
                else:
                    self.raw_list[i].append(self.out_list[i][j])

    def back_function(self):
        self.hide()
        self.screen = fire_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def transfer_function(self):
        self.set_loci[1] = [5, 0, 10, 5, 45, 1.25, 0.1, 90, 80, 0]
        self.hide()
        self.screen = fire_setting_uni(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def skip_function(self):
        num_fire_front_list = []
        fire_delay_list = []
        fuel_coef_list = []
        wind_speed_list = []
        wind_direction_list = []

        for i in range(self.environment_para[2]):
            num_fire_front_list.append(5)
            fire_delay_list.append(0)
            fuel_coef_list.append(10)
            wind_speed_list.append(5)
            wind_direction_list.append(45)

        self.set_loci[1] = [num_fire_front_list, fire_delay_list, fuel_coef_list, wind_speed_list, wind_direction_list, 1.25, 0.1, 90, 80, 1]
        self.hide()
        self.screen = agent_base_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def next_function(self):
        if self.condition_check():
            self.set_loci[1] = self.raw_list
            self.set_loci[1][5] = self.raw_list[5][0] + self.raw_list[5][1]
            self.set_loci[1][6] = self.raw_list[6][0] + self.raw_list[6][1]
            self.hide()
            self.screen = agent_base_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
            self.screen.show()

# Agent base setting
class agent_base_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(agent_base_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[2]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.update_flag = 0
        self.global_applied_flag = applied_flag
        self.orient_flag = set_Loci[2][1]
        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Agent Base Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = ['']
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("Agent_Base_Setting")
        self.setting_init.setText("Agent Base Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Agent Bases:", "num_ageent_bases"), str(1))

        self.agent_base_orient = QLabel(self)
        self.agent_base_orient.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.agent_base_orient.setFont(font)
        self.agent_base_orient.setObjectName('Agent Base Orientation')
        self.agent_base_orient.setText('The Agent Base Orientation: ')

        self.agent_base_orient_horizon = QRadioButton('Horizontal, A 2 × 4 Grid will be Marked',self)
        self.agent_base_orient_horizon.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, 380, 30)
        self.agent_base_orient_horizon.setFont(font)
        self.agent_base_orient_horizon.setObjectName("Agent_Base_Horizon")

        self.agent_base_orient_vertical = QRadioButton('Vertical, A 4 × 2 Grid will be Marked', self)
        self.agent_base_orient_vertical.setGeometry(self.left_upper_x, self.left_upper_y + 4.5 * self.left_y_interval, 380, 30)
        self.agent_base_orient_vertical.setFont(font)
        self.agent_base_orient_vertical.setObjectName("Agent_Base_Vertical")

        self.buttongroup = QButtonGroup(self)
        self.buttongroup.addButton(self.agent_base_orient_horizon, 11)
        self.buttongroup.addButton(self.agent_base_orient_vertical, 12)
        self.buttongroup.buttonClicked.connect(self.buttonclicked)
        if self.orient_flag == 1:
            self.agent_base_orient_vertical.setChecked(True)
            self.agent_base_orient_horizon.setChecked(False)
        else:
            self.agent_base_orient_vertical.setChecked(False)
            self.agent_base_orient_horizon.setChecked(True)

        self.agent_base_orient = QLabel(self)
        self.agent_base_orient.setGeometry(self.left_upper_x, self.left_upper_y + 5.5 * self.left_y_interval, 260, 30)
        self.agent_base_orient.setFont(font_Bold)
        self.agent_base_orient.setObjectName('Agent Base Orientation')
        self.agent_base_orient.setText('Current Orientation: ')

        if self.orient_flag == 1:
            self.agent_base_orient_Display = QLabel(self)
            self.agent_base_orient_Display.setGeometry(self.left_upper_x + 280, self.left_upper_y + 5.5 * self.left_y_interval, 120, 30)
            self.agent_base_orient_Display.setFont(font)
            self.agent_base_orient_Display.setObjectName('Agent Base Orientation Display')
            self.agent_base_orient_Display.setText('Vertical')

            self.size_note = QLabel(self)
            self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 6.5 * self.left_y_interval, 60, 30)
            self.size_note.setFont(font_Bold)
            self.size_note.setObjectName('Note1')
            self.size_note.setText('Note: ')

            self.note = QLabel(self)
            self.note.setGeometry(self.left_upper_x + 60, self.left_upper_y + 6.65 * self.left_y_interval,
                                  400, self.text_height)
            self.note.setFont(font)
            self.note.setObjectName("Note2")
            self.note.setText("You can only pose the entire agent base \nwithin A and " + self.x_coord_list[self.grid_size - 2])
        else:
            self.agent_base_orient_Display = QLabel(self)
            self.agent_base_orient_Display.setGeometry(self.left_upper_x + 280,
                                                       self.left_upper_y + 5.5 * self.left_y_interval, 120, 30)
            self.agent_base_orient_Display.setFont(font)
            self.agent_base_orient_Display.setObjectName('Agent Base Orientation Display')
            self.agent_base_orient_Display.setText('Horizontal')

            self.size_note = QLabel(self)
            self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 6.5 * self.left_y_interval, 60, 30)
            self.size_note.setFont(font_Bold)
            self.size_note.setObjectName('Note1')
            self.size_note.setText('Note: ')

            self.note = QLabel(self)
            self.note.setGeometry(self.left_upper_x + 60, self.left_upper_y + 6.65 * self.left_y_interval,
                                  400, self.text_height)
            self.note.setFont(font)
            self.note.setObjectName("Note2")
            self.note.setText("You can only pose the entire agent base \nwithin 1 and " + str(self.grid_size - 1))

        self.location_init = QLabel(self)
        self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 8 * self.left_y_interval, self.text_display_len, self.text_height)
        self.location_init.setFont(font_Bold)
        self.location_init.setObjectName("agent_base_location")
        self.location_init.setText("Agent Base Locations:")

        for i in range(1):
            # Initialize fire center input controller
            environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (9 + i) * self.left_y_interval),
                                    ("Agent Base #" + str(i + 1) + ": ", "agent_base" + str(i + 1) + "_center"), i)

        self.word_plot(font_grid)
        self.set_initial_value_function()
        self.applied_flag = self.global_applied_flag[2]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 2:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(255, 225, 0)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def buttonclicked(self):
        font = QFont('arial')
        font.setPointSize(14)

        if self.buttongroup.checkedId() == 11:
            self.orient_flag = 0
            self.agent_base_orient_Display.setText('Horizontal')

            self.note.setText(
                "You can only pose the entire agent base \nwithin 1 and " + str(self.grid_size - 1))
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)
        else:
            self.orient_flag = 1
            self.agent_base_orient_Display.setText('Vertical')

            self.note.setText(
                "You can only pose the entire agent base \nwithin A and " + self.x_coord_list[self.grid_size - 2])
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def agent_base_location_determination(self, index):
        default = False
        raw_coord_init = self.raw_list[index].split('-')
        if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
            QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
        else:
            if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
            else:
                if (raw_coord_init[0] > self.x_coord_list[self.grid_size - 1]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size) or (int(raw_coord_init[1]) < 1):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                else:
                    raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                    if (((raw_coord[0] == (self.grid_size - 1)) or (raw_coord[1] >= (self.grid_size - 2))) and (self.orient_flag == 1)) or (((raw_coord[1] == (self.grid_size)) or (raw_coord[0] >= self.grid_size - 3)) and (self.orient_flag == 0)):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Part of the agent base exceeds the world range')
                    elif not self.overlap_check(raw_coord[0], raw_coord[1] - 1)  and self.applied_flag == 0:
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Agent base overlaps with existing objects')
                    else:
                        if self.orient_flag == 1:
                            if (raw_coord[0] != 0) and (raw_coord[0] != (self.grid_size - 2)):
                                QMessageBox.warning(self, 'Warning',
                                                    'Invalid input: Please follow the instruction on the agent base position')
                            else:
                                default = True
                        else:
                            if (raw_coord[1] != 1) and (raw_coord[1] != (self.grid_size - 1)):
                                QMessageBox.warning(self, 'Warning',
                                                    'Invalid input: Please follow the instruction on the agent base position')
                            else:
                                default = True
        return default

    def array_to_string(self, list):
        if self.orient_flag == 1:
            raw_list = [round(list[0]//100) - 1, round(list[1]//100) - 1]
        else:
            raw_list = [round(list[0]//100) - 2, round(list[1]//100)]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        if len(self.out_list[0]) == 0:
            self.raw_list[0] = ''
        else:
            self.raw_list[0] = self.array_to_string(self.out_list[0])

    def set_initial_value_function(self):
        self.agent_base1_center_Edit.setText(self.raw_list[0])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.agent_base1_center_Edit.clear()
        self.raw_list[0] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if self.orient_flag == 1:
            if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and \
               (self.current_grid[pos_x][pos_y + 1] == 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0) and \
                (self.current_grid[pos_x][pos_y + 2] == 0) and (self.current_grid[pos_x + 1][pos_y + 2] == 0) and \
                (self.current_grid[pos_x][pos_y + 3] == 0) and (self.current_grid[pos_x + 1][pos_y + 3] == 0):
                default = True
        else:
            if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and \
                    (self.current_grid[pos_x + 2][pos_y] == 0) and (self.current_grid[pos_x + 3][pos_y] == 0) and \
                    (self.current_grid[pos_x][pos_y + 1] == 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0) and \
                    (self.current_grid[pos_x + 2][pos_y + 1] == 0) and (self.current_grid[pos_x + 3][pos_y + 1] == 0):
                default = True

        return default

    def apply_function(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 2:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 2:
                    self.temp_grid[i][j] = 0

        determine_flag = True
        for i in range(1):
            if (not self.agent_base_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                if self.orient_flag == 1:
                    for i1 in range(2):
                        for j1 in range(4):
                            self.temp_grid[self.x_coord_list.index(raw_coord[0]) + i1][
                                int(raw_coord[1]) + j1 - 1] = 2
                else:
                    for i1 in range(4):
                        for j1 in range(2):
                            self.temp_grid[self.x_coord_list.index(raw_coord[0]) + i1][
                                int(raw_coord[1]) + j1 - 1] = 2

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[2] = [[], 1]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 2:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[2] = 0
        self.hide()
        if self.set_Loci[1][9] == 1:
            self.screen = fire_setting_spec(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
        else:
            self.screen = fire_setting_uni(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each agent base first')
        else:
            determine_flag = True
            for i in range(1):
                if (not self.agent_base_location_determination(i)):
                    determine_flag = False

            if determine_flag:
                new_list = []
                for i in range(1):
                    raw_coord = self.raw_list[i].split('-')

                    if self.orient_flag == 1:
                        new_list = [(self.x_coord_list.index(raw_coord[0]) + 1) * 100, (int(raw_coord[1]) + 1) * 100]
                        for i1 in range(2):
                            for j1 in range(4):
                                self.current_grid[self.x_coord_list.index(raw_coord[0]) + i1][int(raw_coord[1]) + j1 - 1] = 2
                    else:
                        new_list = [(self.x_coord_list.index(raw_coord[0]) + 2) * 100, (int(raw_coord[1])) * 100]
                        for i1 in range(4):
                            for j1 in range(2):
                                self.current_grid[self.x_coord_list.index(raw_coord[0]) + i1][int(raw_coord[1]) + j1 - 1] = 2

                self.set_Loci[2] = [new_list, self.orient_flag]
                self.global_applied_flag[2] = self.applied_flag
                self.hide()
                self.screen = house_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
                self.screen.show()

# House setting
class house_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(house_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[3]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.update_flag = 0
        self.global_applied_flag = applied_flag

        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('House Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = []
        for i in range(self.environment_para[3]):
            self.raw_list.append('')
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("House_Setting")
        self.setting_init.setText("House Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Houses:", "num_houses"), str(self.environment_para[3]))

        self.size_note = QLabel(self)
        self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 60, 30)
        self.size_note.setFont(font_Bold)
        self.size_note.setObjectName('Note')
        self.size_note.setText('Note: ')

        self.house_size = QLabel(self)
        self.house_size.setGeometry(self.left_upper_x + 60, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.house_size.setFont(font)
        self.house_size.setObjectName('House Size')
        self.house_size.setText('A 2 × 2 Grid will be Marked')

        if self.environment_para[3] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_house")
            self.location_init.setText("No House in this Environment")
        else:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len, self.text_height)
            self.location_init.setFont(font_Bold)
            self.location_init.setObjectName("house_location")
            self.location_init.setText("House Locations:")

            for i in range(self.environment_para[3]):
                # Initialize fire center input controller
                environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (4.5 + i) * self.left_y_interval),
                                        ("House #" + str(i + 1) + ": ", "house" + str(i + 1) + "_center"), i)
        self.word_plot(font_grid)
        self.set_initial_value_function()

        self.applied_flag = self.global_applied_flag[3]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 3:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(255, 165, 0)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def house_location_determination(self, index):
        default = False
        if self.environment_para[3] > 0:
            raw_coord_init = self.raw_list[index].split('-')
            if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
            else:
                if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
                else:
                    if (raw_coord_init[0] > self.x_coord_list[self.grid_size - 1]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size) or (int(raw_coord_init[1]) < 1):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                    else:
                        raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                        if (raw_coord[0] == (self.grid_size - 1)) or (raw_coord[1] == self.grid_size):
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Part of the houses exceeds the world range')
                        elif not self.overlap_check(raw_coord[0], raw_coord[1] - 1) and self.applied_flag == 0:
                            QMessageBox.warning(self, 'Warning', 'Invalid input: House ' + str(index + 1) + ' overlaps with existing objects')
                        else:
                            default = True
        return default

    def array_to_string(self, list):
        raw_list = [round(list[0]//100) - 1, round(list[1]//100)]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.out_list[i])

    def set_initial_value_function(self):
        if self.environment_para[3] > 0:
            self.house1_center_Edit.setText(self.raw_list[0])
            if self.environment_para[3] > 1:
                self.house2_center_Edit.setText(self.raw_list[1])
            if self.environment_para[3] > 2:
                self.house3_center_Edit.setText(self.raw_list[2])
            if self.environment_para[3] > 3:
                self.house4_center_Edit.setText(self.raw_list[3])
            if self.environment_para[3] > 4:
                self.house5_center_Edit.setText(self.raw_list[4])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        if self.environment_para[3] > 0:
            self.house1_center_Edit.clear()
            self.raw_list[0] = ''
            if self.environment_para[3] > 1:
                self.house2_center_Edit.clear()
                self.raw_list[1] = ''
            if self.environment_para[3] > 2:
                self.house3_center_Edit.clear()
                self.raw_list[2] = ''
            if self.environment_para[3] > 3:
                self.house4_center_Edit.clear()
                self.raw_list[3] = ''
            if self.environment_para[3] > 4:
                self.house5_center_Edit.clear()
                self.raw_list[4] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and (
                self.current_grid[pos_x][pos_y + 1]== 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0):
            if (self.temp_grid[pos_x][pos_y] == 0) and (self.temp_grid[pos_x + 1][pos_y] == 0) and (
                    self.temp_grid[pos_x][pos_y + 1] == 0) and (self.temp_grid[pos_x + 1][pos_y + 1] == 0):
                default = True
        return default

    def apply_function(self):
        determine_flag = True
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 3:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 3:
                    self.temp_grid[i][j] = 0

        for i in range(self.environment_para[3]):
            if (self.environment_para[3] > i) and (not self.house_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                break

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 3
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 3
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 3
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 3

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[3] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 3:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[3] = 0
        self.hide()
        self.screen = agent_base_location_define(self.environment_para, self.robo_team_para, self.set_Loci,
                                                   self.current_grid, self.global_applied_flag)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each house first')
        else:
            determine_flag = True
            for i in range(self.environment_para[3]):
                if (self.environment_para[3] > i) and (not self.house_location_determination(i)):
                    determine_flag = False
                    self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                    break
            if determine_flag:
                new_list = []
                for i in range(self.environment_para[3]):
                    raw_coord = self.raw_list[i].split('-')
                    new_list.append([(self.x_coord_list.index(raw_coord[0]) + 1) * 100, int(raw_coord[1]) * 100])
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 3
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 3
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 3
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 3

                self.set_Loci[3] = new_list
                self.global_applied_flag[3] = self.applied_flag
                self.hide()
                self.screen = hospital_location_define(self.environment_para, self.robo_team_para, self.set_Loci,
                                                       self.current_grid, self.global_applied_flag)
                self.screen.show()

# Hospital setting
class hospital_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(hospital_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[4]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.global_applied_flag = applied_flag
        #self.applied_flag = applied_flag[4]
        self.update_flag = 0
        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Hospital Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = []
        for i in range(self.environment_para[4]):
            self.raw_list.append('')
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("Hospital_Setting")
        self.setting_init.setText("Hospital Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Hosptials:", "num_hospitals"), str(self.environment_para[4]))

        self.size_note = QLabel(self)
        self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 60, 30)
        self.size_note.setFont(font_Bold)
        self.size_note.setObjectName('Note')
        self.size_note.setText('Note: ')

        self.hospital_size = QLabel(self)
        self.hospital_size.setGeometry(self.left_upper_x + 60, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.hospital_size.setFont(font)
        self.hospital_size.setObjectName('Hospital Size')
        self.hospital_size.setText('A 2 × 2 Grid will be Marked')

        if self.environment_para[4] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Hospital")
            self.location_init.setText("No Hospital in this Environment")
        else:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len, self.text_height)
            self.location_init.setFont(font_Bold)
            self.location_init.setObjectName("hospital_location")
            self.location_init.setText("Hospital Locations:")

            for i in range(self.environment_para[4]):
                # Initialize fire center input controller
                environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (4.5 + i) * self.left_y_interval),
                                        ("Hospital #" + str(i + 1) + ": ", "hospital" + str(i + 1) + "_center"), i)

        self.word_plot(font_grid)
        self.set_initial_value_function()
        self.applied_flag = self.global_applied_flag[4]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 4:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(255, 255, 255)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def hospital_location_determination(self, index):
        default = False
        if self.environment_para[4] > 0:
            raw_coord_init = self.raw_list[index].split('-')
            if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
            else:
                if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
                else:
                    if (raw_coord_init[0] > self.x_coord_list[self.grid_size - 1]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size) or (int(raw_coord_init[1]) < 1):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                    else:
                        raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                        if (raw_coord[0] == (self.grid_size - 1)) or (raw_coord[1] == self.grid_size):
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Part of the hospitals exceeds the world range')
                        elif not self.overlap_check(raw_coord[0], raw_coord[1] - 1) and self.applied_flag == 0:
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Hospital ' + str(index + 1) + ' overlaps with existing objects')
                        else:
                            default = True
        return default

    def array_to_string(self, list):
        raw_list = [round(list[0]//100) - 1, round(list[1]//100)]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.out_list[i])

    def set_initial_value_function(self):
        if self.environment_para[4] > 0:
            self.hospital1_center_Edit.setText(self.raw_list[0])
            if self.environment_para[4] > 1:
                self.hospital2_center_Edit.setText(self.raw_list[1])
            if self.environment_para[4] > 2:
                self.hospital3_center_Edit.setText(self.raw_list[2])
            if self.environment_para[4] > 3:
                self.hospital4_center_Edit.setText(self.raw_list[3])
            if self.environment_para[4] > 4:
                self.hospital5_center_Edit.setText(self.raw_list[4])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        if self.environment_para[4] > 0:
            self.hospital1_center_Edit.clear()
            self.raw_list[0] = ''
            if self.environment_para[4] > 1:
                self.hospital2_center_Edit.clear()
                self.raw_list[1] = ''
            if self.environment_para[4] > 2:
                self.hospital3_center_Edit.clear()
                self.raw_list[2] = ''
            if self.environment_para[4] > 3:
                self.hospital4_center_Edit.clear()
                self.raw_list[3] = ''
            if self.environment_para[4] > 4:
                self.hospital5_center_Edit.clear()
                self.raw_list[4] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and (
                self.current_grid[pos_x][pos_y + 1]== 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0):
            if (self.temp_grid[pos_x][pos_y] == 0) and (self.temp_grid[pos_x + 1][pos_y] == 0) and (
                    self.temp_grid[pos_x][pos_y + 1] == 0) and (self.temp_grid[pos_x + 1][pos_y + 1] == 0):
                default = True
        return default

    def apply_function(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 4:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 4:
                    self.temp_grid[i][j] = 0

        determine_flag = True
        for i in range(self.environment_para[4]):
            if (self.environment_para[4] > i) and (not self.hospital_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                break

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 4
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 4
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 4
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 4

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[4] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 4:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[4] = 0
        self.hide()
        self.screen = house_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each hospital first')
        else:
            determine_flag = True
            for i in range(self.environment_para[4]):
                if (self.environment_para[4] > i) and (not self.hospital_location_determination(i)):
                    determine_flag = False
                    break

            if determine_flag:
                new_list = []
                for i in range(self.environment_para[4]):
                    raw_coord = self.raw_list[i].split('-')
                    new_list.append([(self.x_coord_list.index(raw_coord[0]) + 1) * 100, int(raw_coord[1]) * 100])
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 4
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 4
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 4
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 4

                self.set_Loci[4] = new_list
                self.global_applied_flag[4] = self.applied_flag
                self.hide()
                self.screen = power_station_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
                self.screen.show()

# The power station setting
class power_station_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(power_station_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[5]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.update_flag = 0
        self.global_applied_flag = applied_flag
        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Power Station Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = []
        for i in range(self.environment_para[5]):
            self.raw_list.append('')
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("Power_Setting")
        self.setting_init.setText("Power Station Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Power Stations:", "num_power_stations"), str(self.environment_para[5]))

        self.size_note = QLabel(self)
        self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 60, 30)
        self.size_note.setFont(font_Bold)
        self.size_note.setObjectName('Note')
        self.size_note.setText('Note: ')

        self.power_station_size = QLabel(self)
        self.power_station_size.setGeometry(self.left_upper_x + 60, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.power_station_size.setFont(font)
        self.power_station_size.setObjectName('Power Station Size')
        self.power_station_size.setText('A 2 × 2 Grid will be Marked')

        if self.environment_para[5] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Power_Station")
            self.location_init.setText("No Power Station in this Environment")
        else:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len, self.text_height)
            self.location_init.setFont(font_Bold)
            self.location_init.setObjectName("power_station_location")
            self.location_init.setText("Power Station Locations:")

            for i in range(self.environment_para[5]):
                # Initialize fire center input controller
                environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (4.5 + i) * self.left_y_interval),
                                        ("Power Station #" + str(i + 1) + ": ", "power_station" + str(i + 1) + "_center"), i)

        self.word_plot(font_grid)
        self.set_initial_value_function()
        self.applied_flag = self.global_applied_flag[5]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 5:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(65, 105, 225)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def power_station_location_determination(self, index):
        default = False
        if self.environment_para[5] > 0:
            raw_coord_init = self.raw_list[index].split('-')
            if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
            else:
                if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
                else:
                    if (raw_coord_init[0] > self.x_coord_list[self.grid_size - 1]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size) or (int(raw_coord_init[1]) < 1):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                    else:
                        raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                        if (raw_coord[0] == (self.grid_size - 1)) or (raw_coord[1] == self.grid_size):
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Part of the power stations exceeds the world range')
                        elif not self.overlap_check(raw_coord[0], raw_coord[1] - 1) and self.applied_flag == 0:
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Power station ' + str(index + 1) + ' overlaps with existing objects')
                        else:
                            default = True
        return default

    def array_to_string(self, list):
        raw_list = [round(list[0]//100) - 1, round(list[1]//100)]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.out_list[i])

    def set_initial_value_function(self):
        if self.environment_para[5] > 0:
            self.power_station1_center_Edit.setText(self.raw_list[0])
            if self.environment_para[5] > 1:
                self.power_station2_center_Edit.setText(self.raw_list[1])
            if self.environment_para[5] > 2:
                self.power_station3_center_Edit.setText(self.raw_list[2])
            if self.environment_para[5] > 3:
                self.power_station4_center_Edit.setText(self.raw_list[3])
            if self.environment_para[5] > 4:
                self.power_station5_center_Edit.setText(self.raw_list[4])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        if self.environment_para[5] > 0:
            self.power_station1_center_Edit.clear()
            self.raw_list[0] = ''
            if self.environment_para[5] > 1:
                self.power_station2_center_Edit.clear()
                self.raw_list[1] = ''
            if self.environment_para[5] > 2:
                self.power_station3_center_Edit.clear()
                self.raw_list[2] = ''
            if self.environment_para[5] > 3:
                self.power_station4_center_Edit.clear()
                self.raw_list[3] = ''
            if self.environment_para[5] > 4:
                self.power_station5_center_Edit.clear()
                self.raw_list[4] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and (
                self.current_grid[pos_x][pos_y + 1]== 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0):
            if (self.temp_grid[pos_x][pos_y] == 0) and (self.temp_grid[pos_x + 1][pos_y] == 0) and (
                    self.temp_grid[pos_x][pos_y + 1] == 0) and (self.temp_grid[pos_x + 1][pos_y + 1] == 0):
                default = True
        return default

    def apply_function(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 5:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 5:
                    self.temp_grid[i][j] = 0

        determine_flag = True
        for i in range(self.environment_para[5]):
            if (self.environment_para[5] > i) and (not self.power_station_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                break

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 5
                self.temp_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 5
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 5
                self.temp_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 5

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[5] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 5:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[5] = 0
        self.hide()
        self.screen = hospital_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each power station first')
        else:
            determine_flag = True
            for i in range(self.environment_para[5]):
                if (self.environment_para[5] > i) and (not self.power_station_location_determination(i)):
                    determine_flag = False
                    break

            if determine_flag:
                new_list = []
                for i in range(self.environment_para[5]):
                    raw_coord = self.raw_list[i].split('-')
                    new_list.append([(self.x_coord_list.index(raw_coord[0]) + 1) * 100, int(raw_coord[1]) * 100])
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1]) - 1] = 5
                    self.current_grid[self.x_coord_list.index(raw_coord[0])][int(raw_coord[1])] = 5
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1]) - 1] = 5
                    self.current_grid[self.x_coord_list.index(raw_coord[0]) + 1][int(raw_coord[1])] = 5

                self.set_Loci[5] = new_list
                self.global_applied_flag[5] = self.applied_flag
                self.hide()
                self.screen = lake_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
                self.screen.show()

# The lake setting page
class lake_location_define(QWidget):
    def __init__(self, environment_para, robo_team_para, set_Loci, current_grid, applied_flag):
        super(lake_location_define, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_Loci = set_Loci
        self.out_list = set_Loci[6]
        self.current_grid = current_grid
        self.grid_size = round(self.environment_para[0]//100)
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        self.update_flag = 0
        self.global_applied_flag = applied_flag
        self.x_coord_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Lake Location')

        self.left_upper_x = 120
        self.left_upper_y = 100
        self.right_upper_x = 640
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 400
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(14)

        font_grid = QFont('arial')
        font_grid.setPointSize(14)

        font_button_small = QFont('arial')
        font_button_small.setPointSize(20)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_Title = QFont('arial', 20, 75)
        font_Bold = QFont('arial', 14, 75)

        self.raw_list = []
        for i in range(self.environment_para[6]):
            self.raw_list.append('')
        self.transmit_value()

        font_pe = QPalette()
        self.applied_display = QLabel(self)
        self.applied_display.setFont(font_Title)
        self.applied_display.setObjectName("Applied_Display")

        self.setting_init = QLabel(self)
        self.setting_init.setGeometry(self.left_upper_x, self.left_upper_y, self.text_display_len, self.text_height)
        self.setting_init.setFont(font_Title)
        self.setting_init.setObjectName("Lake_Setting")
        self.setting_init.setText("Lake Setting:")

        # House number input controller
        parameter_Display_Ctl(self, font, font_Bold, (self.left_upper_x, self.left_upper_y + 1.5 * self.left_y_interval), ("Number of Lakes:", "num_lakes"), str(self.environment_para[6]))

        self.size_note = QLabel(self)
        self.size_note.setGeometry(self.left_upper_x, self.left_upper_y + 2.5 * self.left_y_interval, 60, 30)
        self.size_note.setFont(font_Bold)
        self.size_note.setObjectName('Note')
        self.size_note.setText('Note: ')

        self.lake_size = QLabel(self)
        self.lake_size.setGeometry(self.left_upper_x + 60, self.left_upper_y + 2.5 * self.left_y_interval, 380, 30)
        self.lake_size.setFont(font)
        self.lake_size.setObjectName('Lake Size')
        self.lake_size.setText('A 4 × 3 Grid will be Marked')

        if self.environment_para[6] == 0:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len,
                                           self.text_height)
            self.location_init.setFont(font)
            self.location_init.setObjectName("No_Lake")
            self.location_init.setText("No Lake in this Environment")
        else:
            self.location_init = QLabel(self)
            self.location_init.setGeometry(self.left_upper_x, self.left_upper_y + 3.5 * self.left_y_interval, self.text_display_len, self.text_height)
            self.location_init.setFont(font_Bold)
            self.location_init.setObjectName("lake_location")
            self.location_init.setText("Lake Locations:")

            for i in range(self.environment_para[6]):
                # Initialize fire center input controller
                environment_Setting_Ctl(self, font, (self.left_upper_x, self.left_upper_y + (4.5 + i) * self.left_y_interval),
                                        ("Lake #" + str(i + 1) + ": ", "lake" + str(i + 1) + "_center"), i)

        self.word_plot(font_grid)
        self.set_initial_value_function()
        self.applied_flag = self.global_applied_flag[6]

        if self.applied_flag == 0:
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
        else:
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')

        self.applied_display.setPalette(font_pe)

        self.reset = QPushButton(self)
        self.reset.setGeometry(220, 730, 200, 50)
        self.reset.setFont(font_button_small)
        self.reset.setText('Reset')
        self.reset.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.reset.clicked.connect(self.reset_function)

        self.apply = QPushButton(self)
        self.apply.setGeometry(220, 800, 200, 50)
        self.apply.setFont(font_button_small)
        self.apply.setText('Apply')
        self.apply.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.apply.clicked.connect(self.apply_function)

        self.back = QPushButton(self)
        self.back.setGeometry(690, 770, 300, 75)
        self.back.setFont(font_button)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1070, 770, 300, 75)
        self.next.setFont(font_button)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def word_plot(self, font):
        # Display the coordinate within the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_word = QLabel(self)
                self.grid_word.setGeometry(1040 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                self.grid_word.setFont(font)
                self.grid_word.setObjectName("Grid_Word")
                self.grid_word.setText(self.x_coord_list[i] + str(round((j + 1)//10)) + str(round((j + 1)%10)))

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(110, 90, 420, 600)

        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(560, 90, 950, 800)

        # Paint the grid (Previous)
        paint_grid = QPainter(self)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 0:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(197, 225, 165)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 1:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 0, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 2:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 225, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 3:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 165, 0)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 4:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(255, 255, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 5:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(65, 105, 225)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
                elif self.current_grid[i][j] == 6:
                    paint_grid.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    paint_grid.setBrush(QBrush(QColor(0, 191, 255)))
                    paint_grid.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)

        if self.update_flag == 1:
            paint_grid2 = QPainter(self)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.temp_grid[i][j] == 6:
                        paint_grid2.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                        paint_grid2.setBrush(QBrush(QColor(0, 191, 255)))
                        paint_grid2.drawRect(1035 + i * 50 - 25 * self.grid_size, 420 + j * 50 - 25 * self.grid_size, 50, 50)
            self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
            self.update_flag = 0

    def lake_location_determination(self, index):
        default = False
        if self.environment_para[6] > 0:
            raw_coord_init = self.raw_list[index].split('-')
            if (raw_coord_init[0] == '') or (raw_coord_init[1] == '') or (len(raw_coord_init[1]) != 2):
                QMessageBox.warning(self, 'Warning', 'Invalid input: Please enter the full coordinates')
            else:
                if (raw_coord_init[0] > 'Z') or (raw_coord_init[0] < 'A'):
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should be capital letter')
                else:
                    if (raw_coord_init[0] > self.x_coord_list[self.grid_size - 3]) or (raw_coord_init[0] < self.x_coord_list[0]) or (int(raw_coord_init[1]) > self.grid_size - 1) or (int(raw_coord_init[1]) < 1):
                        QMessageBox.warning(self, 'Warning', 'Invalid input: Coordinates should locate within the grid range (Now is ' + str(self.grid_size) + ')')
                    else:
                        raw_coord = [self.x_coord_list.index(raw_coord_init[0]), int(raw_coord_init[1])]
                        if (raw_coord[0] >= (self.grid_size - 2)) or (raw_coord[1] >= (self.grid_size - 2)):
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Part of the lakes exceeds the world range')
                        elif not self.overlap_check(raw_coord[0], raw_coord[1] - 1) and self.applied_flag == 0:
                            QMessageBox.warning(self, 'Warning', 'Invalid input: Lake ' + str(index + 1) + ' overlaps with existing objects')
                        else:
                            default = True

        return default

    def array_to_string(self, list):
        raw_list = [round((list[0] - 50)//100) - 1, round(list[1]//100) - 1]
        out_str = str(self.x_coord_list[raw_list[0]]) + '-' + str(raw_list[1]//10) + str(raw_list[1]%10)
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.out_list) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.out_list[i])

    def set_initial_value_function(self):
        if self.environment_para[6] > 0:
            self.lake1_center_Edit.setText(self.raw_list[0])
            if self.environment_para[6] > 1:
                self.lake2_center_Edit.setText(self.raw_list[1])
            if self.environment_para[6] > 2:
                self.lake3_center_Edit.setText(self.raw_list[2])
            if self.environment_para[6] > 3:
                self.lake4_center_Edit.setText(self.raw_list[3])
            if self.environment_para[6] > 4:
                self.lake5_center_Edit.setText(self.raw_list[4])

    def reset_function(self):
        self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
        if self.environment_para[6] > 0:
            self.lake1_center_Edit.clear()
            self.raw_list[0] = ''
            if self.environment_para[6] > 1:
                self.lake2_center_Edit.clear()
                self.raw_list[1] = ''
            if self.environment_para[6] > 2:
                self.lake3_center_Edit.clear()
                self.raw_list[2] = ''
            if self.environment_para[6] > 3:
                self.lake4_center_Edit.clear()
                self.raw_list[3] = ''
            if self.environment_para[6] > 4:
                self.lake5_center_Edit.clear()
                self.raw_list[4] = ''
        self.update_flag = 1
        self.update()

    def overlap_check(self, pos_x, pos_y):
        default = False
        if (self.current_grid[pos_x][pos_y] == 0) and (self.current_grid[pos_x + 1][pos_y] == 0) and (self.current_grid[pos_x + 2][pos_y] == 0) and \
           (self.current_grid[pos_x][pos_y + 1] == 0) and (self.current_grid[pos_x + 1][pos_y + 1] == 0) and ( self.current_grid[pos_x + 2][pos_y + 1] == 0) and \
            (self.current_grid[pos_x][pos_y + 2] == 0) and (self.current_grid[pos_x + 1][pos_y + 2] == 0) and ( self.current_grid[pos_x + 2][pos_y + 2] == 0) and \
            (self.current_grid[pos_x][pos_y + 3] == 0) and (self.current_grid[pos_x + 1][pos_y + 3] == 0) and ( self.current_grid[pos_x + 2][pos_y + 3] == 0):
            if (self.temp_grid[pos_x][pos_y] == 0) and (self.temp_grid[pos_x + 1][pos_y] == 0) and (self.temp_grid[pos_x + 2][pos_y] == 0) and \
            (self.temp_grid[pos_x][pos_y + 1] == 0) and (self.temp_grid[pos_x + 1][pos_y + 1] == 0) and (self.temp_grid[pos_x + 2][pos_y + 1] == 0) and \
            (self.temp_grid[pos_x][pos_y + 2] == 0) and (self.temp_grid[pos_x + 1][pos_y + 2] == 0) and (self.temp_grid[pos_x + 2][pos_y + 2] == 0) and \
            (self.temp_grid[pos_x][pos_y + 3] == 0) and (self.temp_grid[pos_x + 1][pos_y + 3] == 0) and (self.temp_grid[pos_x + 2][pos_y + 3] == 0):
                default = True
        return default

    def apply_function(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 6:
                    self.current_grid[i][j] = 0
                if self.temp_grid[i][j] == 6:
                    self.temp_grid[i][j] = 0

        determine_flag = True
        for i in range(self.environment_para[6]):
            if (self.environment_para[6] > i) and (not self.lake_location_determination(i)):
                determine_flag = False
                self.temp_grid = list(np.zeros((self.grid_size, self.grid_size), dtype=float))
                break

            if determine_flag:
                raw_coord = self.raw_list[i].split('-')
                for i1 in range(3):
                    for j1 in range(4):
                        self.temp_grid[self.x_coord_list.index(raw_coord[0]) + i1][int(raw_coord[1]) + j1 - 1] = 6

        if determine_flag:
            self.update_flag = 1
            self.applied_flag = 1
            font_pe = QPalette()
            self.applied_display.setGeometry(270, 650, 100, 40)
            font_pe.setColor(QPalette.WindowText, QColor(60,179,113))
            self.applied_display.setText('Applied')
            self.applied_display.setPalette(font_pe)
            self.update()
        else:
            self.applied_flag = 0
            font_pe = QPalette()
            self.applied_display.setGeometry(240, 650, 160, 40)
            font_pe.setColor(QPalette.WindowText, Qt.red)
            self.applied_display.setText('Not Applied')
            self.applied_display.setPalette(font_pe)

    def back_function(self):
        self.set_Loci[6] = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.current_grid[i][j] == 6:
                    self.current_grid[i][j] = 0
        self.global_applied_flag[6] = 0
        self.hide()
        self.screen = power_station_location_define(self.environment_para, self.robo_team_para, self.set_Loci, self.current_grid, self.global_applied_flag)
        self.screen.show()

    def next_function(self):
        if self.applied_flag == 0:
            QMessageBox.warning(self, 'Warning', 'Please view the location of each lake first')
        else:
            determine_flag = True
            for i in range(self.environment_para[6]):
                if (self.environment_para[6] > i) and (not self.lake_location_determination(i)):
                    determine_flag = False

            if determine_flag:
                new_list = []
                for i in range(self.environment_para[6]):
                    raw_coord = self.raw_list[i].split('-')
                    new_list.append([(self.x_coord_list.index(raw_coord[0]) + 1) * 100 + 50, (int(raw_coord[1]) + 1) * 100])
                    for i1 in range(3):
                        for j1 in range(4):
                            self.current_grid[self.x_coord_list.index(raw_coord[0]) + i1][int(raw_coord[1]) + j1 - 1] = 6

                self.set_Loci[6] = new_list
                self.global_applied_flag[6] = self.applied_flag
                self.hide()
                adv_setting_list = [[], [], [], [], [], [], [], [], [], []]
                if self.robo_team_para[3] == 0:
                    self.screen = homo_adv_setting(self.environment_para, self.robo_team_para, self.set_Loci, adv_setting_list, self.current_grid, self.global_applied_flag)
                else:
                    self.screen = hetero_adv_setting(self.environment_para, self.robo_team_para, self.set_Loci, adv_setting_list, self.current_grid, self.global_applied_flag)
                self.screen.show()

# The homogeneous agent setting page
class homo_adv_setting(QWidget):
    def __init__(self, environment_para, robo_team_para, set_loci, adv_setting_list, current_grid, applied_flag):
        super(homo_adv_setting, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_loci
        self.adv_setting_list = adv_setting_list
        self.current_grid = current_grid
        self.applied_flag = applied_flag
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Advanced Setting (Homogeneous)')

        self.right_upper_x = 160
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 500
        self.text_height = 40

        self.raw_list = ['', '', '', '', '', '', '', '', '', '']
        self.transmit_value()
        self.set_default()

        font = QFont('arial')
        font.setPointSize(14)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button2 = QFont('arial')
        font_button2.setPointSize(20)

        font_Title = QFont('arial', 20, 75)

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.right_upper_x, self.right_upper_y, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("adv_setting_Homo")
        self.environment_setup.setText("Advanced Setting (Homogeneous):")

        # Perception Agent altitude input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 1.5 * self.left_y_interval), ("Perception Agent Altitude Limitations:", "perception_agent_altitude"), 1)
        if len(self.adv_setting_list[0]) == 0:
            self.perception_agent_altitude_Edit.setPlaceholderText('[𝑧_𝑚𝑖𝑛,𝑧_𝑚𝑎𝑥]')
        else:
            self.perception_agent_altitude_Edit.setText(self.array_to_string(self.adv_setting_list[0]))
        self.perception_agent_altitude_Edit.textChanged.connect(lambda:self.perception_agent_altitude_function((self.perception_agent_altitude_Edit.text())))

        # Hybrid Agent altitude input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval), ("Hybrid Agent Altitude Limitations:", "hybrid_agent_altitude"), 2)
        if len(self.adv_setting_list[1]) == 0:
            self.hybrid_agent_altitude_Edit.setPlaceholderText('[𝑧_𝑚𝑖𝑛,𝑧_𝑚𝑎𝑥]')
        else:
            self.hybrid_agent_altitude_Edit.setText(self.array_to_string(self.adv_setting_list[1]))
        self.hybrid_agent_altitude_Edit.textChanged.connect(lambda:self.hybrid_agent_altitude_function((self.hybrid_agent_altitude_Edit.text())))

        # Perception Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 4 * self.left_y_interval), ("Perception Agents Battery Limit:", "Perception_Agents_Battery_Limit"), 3)
        if len(self.adv_setting_list[2]) == 0:
            self.Perception_Agents_Battery_Limit_Edit.setPlaceholderText('𝑏_𝑚𝑎𝑥')
        else:
            self.Perception_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[2]))
        self.Perception_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Perception_Agents_Battery_Limit_function((self.Perception_Agents_Battery_Limit_Edit.text())))

        # Action Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 5 * self.left_y_interval), ("Action Agents Battery Limit:", "Action_Agents_Battery_Limit"), 4)
        if len(self.adv_setting_list[3]) == 0:
            self.Action_Agents_Battery_Limit_Edit.setPlaceholderText('𝑏_𝑚𝑎𝑥')
        else:
            self.Action_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[3]))
        self.Action_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Action_Agents_Battery_Limit_function((self.Action_Agents_Battery_Limit_Edit.text())))

        # Hybrid Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 6 * self.left_y_interval), ("Hybrid Agents Battery Limit:", "Hybrid_Agents_Battery_Limit"), 5)
        if len(self.adv_setting_list[4]) == 0:
            self.Hybrid_Agents_Battery_Limit_Edit.setPlaceholderText('𝑏_𝑚𝑎𝑥')
        else:
            self.Hybrid_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[4]))
        self.Hybrid_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Battery_Limit_function((self.Hybrid_Agents_Battery_Limit_Edit.text())))

        # Perception Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 7.5 * self.left_y_interval), ("Perception Agents Velocity Limit:", "Perception_Agents_Velocity_Limit"), 6)
        if len(self.adv_setting_list[5]) == 0:
            self.Perception_Agents_Velocity_Limit_Edit.setPlaceholderText('𝑣_𝑚𝑎𝑥')
        else:
            self.Perception_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[5]))
        self.Perception_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Perception_Agents_Velocity_Limit_function((self.Perception_Agents_Velocity_Limit_Edit.text())))

        # Action Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 8.5 * self.left_y_interval), ("Action Agents Velocity Limit:", "Action_Agents_Velocity_Limit"), 7)
        if len(self.adv_setting_list[6]) == 0:
            self.Action_Agents_Velocity_Limit_Edit.setPlaceholderText('𝑣_𝑚𝑎𝑥')
        else:
            self.Action_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[6]))
        self.Action_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Action_Agents_Velocity_Limit_function((self.Action_Agents_Velocity_Limit_Edit.text())))

        # Hybrid Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 9.5 * self.left_y_interval), ("Hybrid Agents Velocity Limit:", "Hybrid_Agents_Velocity_Limit"), 8)
        if len(self.adv_setting_list[7]) == 0:
            self.Hybrid_Agents_Velocity_Limit_Edit.setPlaceholderText('𝑣_𝑚𝑎𝑥')
        else:
            self.Hybrid_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[7]))
        self.Hybrid_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Velocity_Limit_function((self.Hybrid_Agents_Velocity_Limit_Edit.text())))

        # Action Agents Tank Capacity input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 11 * self.left_y_interval), ("Action Agents Tank Capacity:", "Action_Agents_Tank_Capacity"), 9)
        if len(self.adv_setting_list[8]) == 0:
            self.Action_Agents_Tank_Capacity_Edit.setPlaceholderText('𝑐_𝑚𝑎𝑥')
        else:
            self.Action_Agents_Tank_Capacity_Edit.setText(self.array_to_string(self.adv_setting_list[8]))
        self.Action_Agents_Tank_Capacity_Edit.textChanged.connect(lambda:self.Action_Agents_Tank_Capacity_Limit_function((self.Action_Agents_Tank_Capacity_Edit.text())))

        # Hybrid Agents Tank Capacity input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 12 * self.left_y_interval), ("Hybrid Agents Tank Capacity:", "Hybrid_Agents_Tank_Capacity"), 10)
        if len(self.adv_setting_list[9]) == 0:
            self.Hybrid_Agents_Tank_Capacity_Edit.setPlaceholderText('𝑐_𝑚𝑎𝑥')
        else:
            self.Hybrid_Agents_Tank_Capacity_Edit.setText(self.array_to_string(self.adv_setting_list[9]))
        self.Hybrid_Agents_Tank_Capacity_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Tank_Capacity_Limit_function((self.Hybrid_Agents_Tank_Capacity_Edit.text())))

        self.back = QPushButton(self)
        self.back.setGeometry(125, 770, 300, 75)
        self.back.setFont(font_button2)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.transfer = QPushButton(self)
        self.transfer.setGeometry(475, 770, 300, 75)
        self.transfer.setFont(font_button2)
        self.transfer.setText('Transfer to \nHeterogenous')
        self.transfer.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.transfer.clicked.connect(self.transfer_function)

        self.skip = QPushButton(self)
        self.skip.setGeometry(825, 770, 300, 75)
        self.skip.setFont(font_button2)
        self.skip.setText('Skip')
        self.skip.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.skip.clicked.connect(self.skip_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1174, 770, 300, 75)
        self.next.setFont(font_button2)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def paintEvent(self, e):
        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(110, 90, 1400, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(140, 110, 1340, 640)

    def set_default(self):
        self.default_list = [[[10, 100]], [[10, 100]], [[500]], [[500]], [[500]], [[20]], [[20]], [[20]], [[10]], [[10]]]

    def perception_agent_altitude_function(self, edit_str):
        self.raw_list[0] = str(edit_str)

    def hybrid_agent_altitude_function(self, edit_str):
        self.raw_list[1] = str(edit_str)

    def Perception_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[2] = str(edit_str)

    def Action_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[3] = str(edit_str)

    def Hybrid_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[4] = str(edit_str)

    def Perception_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[5] = str(edit_str)

    def Action_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[6] = str(edit_str)

    def Hybrid_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[7] = str(edit_str)

    def Action_Agents_Tank_Capacity_Limit_function(self, edit_str):
        self.raw_list[8] = str(edit_str)

    def Hybrid_Agents_Tank_Capacity_Limit_function(self, edit_str):
        self.raw_list[9] = str(edit_str)

    def perception_agent_altitude_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[0] = []
            return True
        else:
            perception_agent_altitude_str = self.raw_list[0].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            perception_agent_altitude_list = perception_agent_altitude_str.split(',')
            if not perception_agent_altitude_str.replace(',', '').isdigit():
                if len(perception_agent_altitude_str)== 0:
                    self.adv_setting_list[0] = self.default_list[0]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agent altitude must be digits')
                    return False
            elif len(perception_agent_altitude_list)!= 2:
                QMessageBox.warning(self, 'Warning', 'The number of the perception agent altitude must be 2 for homogeneous team')
                return False
            else:
                height_flag = digit_validate_double(perception_agent_altitude_list, 10, -1, 100, 1)
                if height_flag == 1:
                    QMessageBox.warning(self, 'Warning',
                                        'The lower bound of the perception agent altitude must between 10 and 100, and lower than the upper bound')
                    return False
                elif height_flag == 2:
                    QMessageBox.warning(self, 'Warning',
                                        'The upper bound of the perception agent altitude must between 10 and 100, and higher than the lower bound')
                    return False
                else:
                    self.adv_setting_list[0] = list_division(perception_agent_altitude_list, 1)
                    return True

    def hybrid_agent_altitude_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[1] = []
            return True
        else:
            hybrid_agent_altitude_str = self.raw_list[1].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            hybrid_agent_altitude_list = hybrid_agent_altitude_str.split(',')
            if not hybrid_agent_altitude_str.replace(',', '').isdigit():
                if len(hybrid_agent_altitude_str)== 0:
                    self.adv_setting_list[1] = self.default_list[1]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agent altitude must be digits')
                    return False
            elif len(hybrid_agent_altitude_list)!= 2:
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agent altitude must be 2 for homogeneous team')
                return False
            else:
                height_flag = digit_validate_double(hybrid_agent_altitude_list, 10, 20, 100, 1)
                if height_flag == 1:
                    QMessageBox.warning(self, 'Warning', 'The lower bound of the hybrid agent altitude must between 10 and 20')
                    return False
                elif height_flag == 2:
                    QMessageBox.warning(self, 'Warning', 'The upper bound of the hybrid agent altitude must between 20 and 100')
                    return False
                else:
                    self.adv_setting_list[1] = list_division(hybrid_agent_altitude_list, 1)
                    return True

    def Perception_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[2] = []
            return True
        else:
            Perception_Agents_Battery_Limit_str = self.raw_list[2].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Perception_Agents_Battery_Limit_list = Perception_Agents_Battery_Limit_str.split(',')
            if not Perception_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Perception_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[2] = self.default_list[2]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agents battery limit must be digits')
                    return False
            elif len(Perception_Agents_Battery_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the perception agents battery limit must be 1 for homogeneous team')
                return False
            else:
                battery_flag = digit_validate(Perception_Agents_Battery_Limit_list, 200, 800, 1)
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Perception agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[2] = list_division_single(Perception_Agents_Battery_Limit_list, 1)
                    return True

    def Action_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[3] = []
            return True
        else:
            Action_Agents_Battery_Limit_str = self.raw_list[3].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Battery_Limit_list = Action_Agents_Battery_Limit_str.split(',')
            if not Action_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[3] = self.default_list[3]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Initial action agent battery limit must be digits')
                    return False
            elif len(Action_Agents_Battery_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the initial action agents battery limit must be 1 for homogeneous team')
                return False
            else:
                battery_flag = digit_validate(Action_Agents_Battery_Limit_list, 200, 800, 1)
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[3] = list_division_single(Action_Agents_Battery_Limit_list, 1)
                    return True

    def Hybrid_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[4] = []
            return True
        else:
            Hybrid_Agents_Battery_Limit_str = self.raw_list[4].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Battery_Limit_list = Hybrid_Agents_Battery_Limit_str.split(',')
            if not Hybrid_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[4] = self.default_list[4]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Initial hybrid agents battery limit must be digits')
                    return False
            elif len(Hybrid_Agents_Battery_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the initial hybrid agents battery limit must be 1 for homogeneous team')
                return False
            else:
                battery_flag = digit_validate(Hybrid_Agents_Battery_Limit_list, 200, 800, 1)
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Hybrid agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[4] = list_division_single(Hybrid_Agents_Battery_Limit_list, 1)
                    return True

    def Perception_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[5] = []
            return True
        else:
            Perception_Agents_Velocity_Limit_str = self.raw_list[5].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Perception_Agents_Velocity_Limit_list = Perception_Agents_Velocity_Limit_str.split(',')
            if not Perception_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Perception_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[5] = self.default_list[5]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agents velocity limit must be digits')
                    return False
            elif len(Perception_Agents_Velocity_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the perception agents velocity limit must be 1 for homogeneous team')
                return False
            else:
                velocity_flag = digit_validate(Perception_Agents_Velocity_Limit_list, 10, 30, 1)
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Perception agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[5] = list_division_single(Perception_Agents_Velocity_Limit_list, 1)
                    return True

    def Action_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[6] = []
            return True
        else:
            Action_Agents_Velocity_Limit_str = self.raw_list[6].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Velocity_Limit_list = Action_Agents_Velocity_Limit_str.split(',')
            if not Action_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[6] = self.default_list[6]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Action agents velocity limit must be digits')
                    return False
            elif len(Action_Agents_Velocity_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the action agents velocity limit must be 1 for homogeneous team')
                return False
            else:
                velocity_flag = digit_validate(Action_Agents_Velocity_Limit_list, 10, 30, 1)
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[6] = list_division_single(Action_Agents_Velocity_Limit_list, 1)
                    return True

    def Hybrid_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[7] = []
            return True
        else:
            Hybrid_Agents_Velocity_Limit_str = self.raw_list[7].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Velocity_Limit_list = Hybrid_Agents_Velocity_Limit_str.split(',')
            if not Hybrid_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[7] = self.default_list[7]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agents velocity limit must be digits')
                    return False
            elif len(Hybrid_Agents_Velocity_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agents velocity limit must be 1 for homogeneous team')
                return False
            else:
                velocity_flag = digit_validate(Hybrid_Agents_Velocity_Limit_list, 10, 30, 1)
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Hybrid agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[7] = list_division_single(Hybrid_Agents_Velocity_Limit_list, 1)
                    return True

    def Action_Agents_Tank_Capacity_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[8] = []
            return True
        else:
            Action_Agents_Tank_Capacity_Limit_str = self.raw_list[8].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Tank_Capacity_Limit_list = Action_Agents_Tank_Capacity_Limit_str.split(',')
            if not Action_Agents_Tank_Capacity_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Tank_Capacity_Limit_str)== 0:
                    self.adv_setting_list[8] = self.default_list[8]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Action agents water tank capacity limit must be digits')
                    return False
            elif len(Action_Agents_Tank_Capacity_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the action agents water tank capacity limit must be 1 for homogeneous team')
                return False
            else:
                water_capacity_flag = digit_validate(Action_Agents_Tank_Capacity_Limit_list, 1, 15, 1)
                if not water_capacity_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents water tank capacity limit must between 1 and 15')
                    return False
                else:
                    self.adv_setting_list[8] = list_division_single(Action_Agents_Tank_Capacity_Limit_list, 1)
                    return True

    def Hybrid_Agents_Tank_Capacity_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[9] = []
            return True
        else:
            Hybrid_Agents_Tank_Capacity_Limit_str = self.raw_list[9].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Tank_Capacity_Limit_list = Hybrid_Agents_Tank_Capacity_Limit_str.split(',')
            if not Hybrid_Agents_Tank_Capacity_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Tank_Capacity_Limit_str)== 0:
                    self.adv_setting_list[9] = self.default_list[9]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: hybrid agents tank capacity must be digits')
                    return False
            elif len(Hybrid_Agents_Tank_Capacity_Limit_list)!= 1:
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agents tank capacity must be 1 for homogeneous team')
                return False
            else:
                water_capacity_flag = digit_validate(Hybrid_Agents_Tank_Capacity_Limit_list, 1, 15, 1)
                if not water_capacity_flag:
                    QMessageBox.warning(self, 'Warning', 'Hybrid agents water tank capacity limit must between 1 and 15')
                    return False
                else:
                    self.adv_setting_list[8] = list_division_single(Hybrid_Agents_Tank_Capacity_Limit_list, 1)
                    return True

    def string_to_array(self):
        default = False
        if self.perception_agent_altitude_Determination():
            if self.hybrid_agent_altitude_Determination():
                if self.Perception_Agents_Battery_Limit_Determination():
                    if self.Action_Agents_Battery_Limit_Determination():
                        if self.Hybrid_Agents_Battery_Limit_Determination():
                            if self.Perception_Agents_Velocity_Limit_Determination():
                                if self.Action_Agents_Velocity_Limit_Determination():
                                    if self.Hybrid_Agents_Velocity_Limit_Determination():
                                        if self.Action_Agents_Tank_Capacity_Limit_Determination():
                                            if self.Hybrid_Agents_Tank_Capacity_Limit_Determination():
                                                default = True
        return default

    def array_to_string(self, list):
        out_str = '['
        for i in range(len(list)):
            if len(list[i]) == 1:
                out_str += str(list[i][0])
            else:
                out_str += '(' + str(list[i][0]) + ', ' + str(list[i][1]) + ')'
            if i < (len(list) - 1):
                out_str += ', '
        out_str += ']'
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.adv_setting_list[i]) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.adv_setting_list[i])

    def back_function(self):
        self.hide()
        self.screen = lake_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def transfer_function(self):
        self.robo_team_para[3] = 1
        self.adv_setting_list = [[], [], [], [], [], [], [], [], [], []]
        self.hide()
        self.screen = hetero_adv_setting(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, self.current_grid, self.applied_flag)
        self.screen.show()

    def skip_function(self):
        self.adv_setting_list = self.default_list
        self.screen = pre_View(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list)
        answer = QMessageBox.question(self, 'Environment Pre-View', "Do you like to proceed with this environment design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.hide()
            self.screen = tutorial(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, 0)
            self.screen.show()

    def next_function(self):
        if self.string_to_array():
            self.screen = pre_View(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list)
            answer = QMessageBox.question(self, 'Environment Pre-View', "Do you like to proceed with this environment design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer == QMessageBox.Yes:
                self.hide()
                self.screen = tutorial(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, 0)
                self.screen.show()

# The heterogeneous agent setting page
class hetero_adv_setting(QWidget):
    def __init__(self, environment_para, robo_team_para, set_loci, adv_setting_list, current_grid, applied_flag):
        super(hetero_adv_setting, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_loci
        self.adv_setting_list = adv_setting_list
        self.current_grid = current_grid
        self.applied_flag = applied_flag
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Advanced Setting (Heterogeneous)')

        self.right_upper_x = 160
        self.right_upper_y = 120

        self.left_y_interval = 40
        self.text_display_len = 500
        self.text_height = 40

        self.raw_list = ['', '', '', '', '', '', '', '', '', '']
        self.transmit_value()
        self.set_default()

        font = QFont('arial')
        font.setPointSize(14)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button2 = QFont('arial')
        font_button2.setPointSize(20)

        font_Title = QFont('arial', 20, 75)

        self.environment_setup = QLabel(self)
        self.environment_setup.setGeometry(self.right_upper_x, self.right_upper_y, self.text_display_len, self.text_height)
        self.environment_setup.setFont(font_Title)
        self.environment_setup.setObjectName("adv_setting_Hetero")
        self.environment_setup.setText("Advanced Setting (Heterogeneous):")

        # Perception Agent altitude input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 1.5 * self.left_y_interval), ("Perception Agent Altitude Limitations:", "perception_agent_altitude"), 1)
        if len(self.adv_setting_list[0]) == 0:
            self.perception_agent_altitude_Edit.setPlaceholderText('[𝑧_𝑚𝑖𝑛,𝑧_𝑚𝑎𝑥]')
        else:
            self.perception_agent_altitude_Edit.setText(self.array_to_string(self.adv_setting_list[0]))
        self.perception_agent_altitude_Edit.textChanged.connect(lambda:self.perception_agents_altitude_function((self.perception_agent_altitude_Edit.text())))

        # Hybrid Agent altitude input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 2.5 * self.left_y_interval), ("Hybrid Agent Altitude Limitations:", "hybrid_agent_altitude"), 2)
        if len(self.adv_setting_list[1]) == 0:
            self.hybrid_agent_altitude_Edit.setPlaceholderText('[𝑧_𝑚𝑖𝑛,𝑧_𝑚𝑎𝑥]')
        else:
            self.hybrid_agent_altitude_Edit.setText(self.array_to_string(self.adv_setting_list[1]))
        self.hybrid_agent_altitude_Edit.textChanged.connect(lambda:self.hybrid_agents_altitude_function((self.hybrid_agent_altitude_Edit.text())))

        # Perception Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 4 * self.left_y_interval), ("Perception Agents Battery Limit:", "Perception_Agents_Battery_Limit"), 3)
        if len(self.adv_setting_list[2]) == 0:
            self.Perception_Agents_Battery_Limit_Edit.setPlaceholderText('[𝑏_𝑚𝑎𝑥^1, …,𝑏_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Perception_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[2]))
        self.Perception_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Perception_Agents_Battery_Limit_function((self.Perception_Agents_Battery_Limit_Edit.text())))

        # Action Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 5 * self.left_y_interval), ("Action Agents Battery Limit:", "Action_Agents_Battery_Limit"), 4)
        if len(self.adv_setting_list[3]) == 0:
            self.Action_Agents_Battery_Limit_Edit.setPlaceholderText('[𝑏_𝑚𝑎𝑥^1, …,𝑏_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Action_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[3]))
        self.Action_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Action_Agents_Battery_Limit_function((self.Action_Agents_Battery_Limit_Edit.text())))

        # Hybrid Agents Battery Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 6 * self.left_y_interval), ("Hybrid Agents Battery Limit:", "Hybrid_Agents_Battery_Limit"), 5)
        if len(self.adv_setting_list[4]) == 0:
            self.Hybrid_Agents_Battery_Limit_Edit.setPlaceholderText('[𝑏_𝑚𝑎𝑥^1, …,𝑏_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Hybrid_Agents_Battery_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[4]))
        self.Hybrid_Agents_Battery_Limit_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Battery_Limit_function((self.Hybrid_Agents_Battery_Limit_Edit.text())))

        # Perception Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 7.5 * self.left_y_interval), ("Perception Agents Velocity Limit:", "Perception_Agents_Velocity_Limit"), 6)
        if len(self.adv_setting_list[5]) == 0:
            self.Perception_Agents_Velocity_Limit_Edit.setPlaceholderText('[v_𝑚𝑎𝑥^1, …,v_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Perception_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[5]))
        self.Perception_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Perception_Agents_Velocity_Limit_function((self.Perception_Agents_Velocity_Limit_Edit.text())))

        # Action Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 8.5 * self.left_y_interval), ("Action Agents Velocity Limit:", "Action_Agents_Velocity_Limit"), 7)
        if len(self.adv_setting_list[6]) == 0:
            self.Action_Agents_Velocity_Limit_Edit.setPlaceholderText('[v_𝑚𝑎𝑥^1, …,v_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Action_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[6]))
        self.Action_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Action_Agents_Velocity_Limit_function((self.Action_Agents_Velocity_Limit_Edit.text())))

        # Hybrid Agents Velocity Limit input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 9.5 * self.left_y_interval), ("Hybrid Agents Velocity Limit:", "Hybrid_Agents_Velocity_Limit"), 8)
        if len(self.adv_setting_list[7]) == 0:
            self.Hybrid_Agents_Velocity_Limit_Edit.setPlaceholderText('[v_𝑚𝑎𝑥^1, …,v_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Hybrid_Agents_Velocity_Limit_Edit.setText(self.array_to_string(self.adv_setting_list[7]))
        self.Hybrid_Agents_Velocity_Limit_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Velocity_Limit_function((self.Hybrid_Agents_Velocity_Limit_Edit.text())))

        # Action Agents Tank Capacity input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 11 * self.left_y_interval), ("Action Agents Tank Capacity:", "Action_Agents_Tank_Capacity"), 9)
        if len(self.adv_setting_list[8]) == 0:
            self.Action_Agents_Tank_Capacity_Edit.setPlaceholderText('[𝑐_𝑚𝑎𝑥^1, …,𝑐_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Action_Agents_Tank_Capacity_Edit.setText(self.array_to_string(self.adv_setting_list[8]))
        self.Action_Agents_Tank_Capacity_Edit.textChanged.connect(lambda:self.Action_Agents_Tank_Capacity_Limit_function((self.Action_Agents_Tank_Capacity_Edit.text())))

        # Hybrid Agent Tank Capacity input controller
        environment_Setting_Ctl_normal(self, font, (self.right_upper_x, self.right_upper_y + 12 * self.left_y_interval), ("Hybrid Agents Tank Capacity:", "Hybrid_Agents_Tank_Capacity"), 10)
        if len(self.adv_setting_list[9]) == 0:
            self.Hybrid_Agents_Tank_Capacity_Edit.setPlaceholderText('[𝑐_𝑚𝑎𝑥^1, …,𝑐_𝑚𝑎𝑥^(𝑁_𝑃 )]')
        else:
            self.Hybrid_Agents_Tank_Capacity_Edit.setText(self.array_to_string(self.adv_setting_list[9]))
        self.Hybrid_Agents_Tank_Capacity_Edit.textChanged.connect(lambda:self.Hybrid_Agents_Tank_Capacity_Limit_function((self.Hybrid_Agents_Tank_Capacity_Edit.text())))

        self.back = QPushButton(self)
        self.back.setGeometry(125, 770, 300, 75)
        self.back.setFont(font_button2)
        self.back.setText('Back')
        self.back.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.back.clicked.connect(self.back_function)

        self.transfer = QPushButton(self)
        self.transfer.setGeometry(475, 770, 300, 75)
        self.transfer.setFont(font_button2)
        self.transfer.setText('Transfer to \nHomgenous')
        self.transfer.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.transfer.clicked.connect(self.transfer_function)

        self.skip = QPushButton(self)
        self.skip.setGeometry(825, 770, 300, 75)
        self.skip.setFont(font_button2)
        self.skip.setText('Skip')
        self.skip.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.skip.clicked.connect(self.skip_function)

        self.next = QPushButton(self)
        self.next.setGeometry(1174, 770, 300, 75)
        self.next.setFont(font_button2)
        self.next.setText('Next >>')
        self.next.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.next.clicked.connect(self.next_function)

    def paintEvent(self, e):
        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(110, 90, 1400, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(140, 110, 1340, 640)

    def perception_agents_altitude_function(self, edit_str):
        self.raw_list[0] = str(edit_str)

    def hybrid_agents_altitude_function(self, edit_str):
        self.raw_list[1] = str(edit_str)

    def Perception_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[2] = str(edit_str)

    def Action_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[3] = str(edit_str)

    def Hybrid_Agents_Battery_Limit_function(self, edit_str):
        self.raw_list[4] = str(edit_str)

    def Perception_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[5] = str(edit_str)

    def Action_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[6] = str(edit_str)

    def Hybrid_Agents_Velocity_Limit_function(self, edit_str):
        self.raw_list[7] = str(edit_str)

    def Action_Agents_Tank_Capacity_Limit_function(self, edit_str):
        self.raw_list[8] = str(edit_str)

    def Hybrid_Agents_Tank_Capacity_Limit_function(self, edit_str):
        self.raw_list[9] = str(edit_str)

    def perception_agent_altitude_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[0] = []
            return True
        else:
            perception_agent_altitude_str = self.raw_list[0].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            perception_agent_altitude_list = perception_agent_altitude_str.split(',')
            if not perception_agent_altitude_str.replace(',', '').isdigit():
                if len(perception_agent_altitude_str)== 0:
                    self.adv_setting_list[0] = self.default_list[0]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agent altitude must be digits')
                    return False
            elif len(perception_agent_altitude_list)!= 2 * self.robo_team_para[0]:
                QMessageBox.warning(self, 'Warning', 'The number of the perception agent altitude must be the same as the robot team parameter')
                return False
            else:
                height_flag = digit_validate_double(perception_agent_altitude_list, 10, -1, 100, self.robo_team_para[0])
                if height_flag == 1:
                    QMessageBox.warning(self, 'Warning',
                                        'The lower bound of the perception agent altitude must between 10 and 100, and lower than the upper bound')
                    return False
                elif height_flag == 2:
                    QMessageBox.warning(self, 'Warning',
                                        'The upper bound of the perception agent altitude must between 10 and 100, and higher than the lower bound')
                    return False
                else:
                    self.adv_setting_list[0] = list_division(perception_agent_altitude_list, self.robo_team_para[0])
                    return True

    def hybrid_agent_altitude_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[1] = []
            return True
        else:
            hybrid_agent_altitude_str = self.raw_list[1].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            hybrid_agent_altitude_list = hybrid_agent_altitude_str.split(',')
            if not hybrid_agent_altitude_str.replace(',', '').isdigit():
                if len(hybrid_agent_altitude_str)== 0:
                    self.adv_setting_list[1] = self.default_list[1]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agent altitude must be digits')
                    return False
            elif len(hybrid_agent_altitude_list)!= 2 * self.robo_team_para[2]:
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agent altitude must be the same as the robot team parameter')
                return False
            else:
                height_flag = digit_validate_double(hybrid_agent_altitude_list, 10, 20, 100, self.robo_team_para[2])
                if height_flag == 1:
                    QMessageBox.warning(self, 'Warning', 'The lower bound of the hybrid agent altitude must between 10 and 20')
                    return False
                elif height_flag == 2:
                    QMessageBox.warning(self, 'Warning', 'The upper bound of the hybrid agent altitude must between 20 and 100')
                    return False
                else:
                    self.adv_setting_list[1] = list_division(hybrid_agent_altitude_list, self.robo_team_para[2])
                    return True

    def Perception_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[2] = []
            return True
        else:
            Perception_Agents_Battery_Limit_str = self.raw_list[2].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Perception_Agents_Battery_Limit_list = Perception_Agents_Battery_Limit_str.split(',')
            if not Perception_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Perception_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[2] = self.default_list[2]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agent battery limit must be digits')
                    return False
            elif len(Perception_Agents_Battery_Limit_list)!= (self.robo_team_para[0]):
                QMessageBox.warning(self, 'Warning', 'The number of the perception agent battery limit must be the same as the robot team parameter')
                return False
            else:
                battery_flag = digit_validate(Perception_Agents_Battery_Limit_list, 200, 800, self.robo_team_para[0])
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Perception agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[2] = list_division_single(Perception_Agents_Battery_Limit_list, self.robo_team_para[0])
                    return True

    def Action_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[3] = []
            return True
        else:
            Action_Agents_Battery_Limit_str = self.raw_list[3].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Battery_Limit_list = Action_Agents_Battery_Limit_str.split(',')
            if not Action_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[3] = self.default_list[3]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Action agent battery limit must be digits')
                    return False
            elif len(Action_Agents_Battery_Limit_list)!= (self.robo_team_para[1]):
                QMessageBox.warning(self, 'Warning', 'The number of the action agent battery limit must be the same as the robot team parameter')
                return False
            else:
                battery_flag = digit_validate(Action_Agents_Battery_Limit_list, 200, 800, self.robo_team_para[1])
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[3] = list_division_single(Action_Agents_Battery_Limit_list, self.robo_team_para[1])
                    return True

    def Hybrid_Agents_Battery_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[4] = []
            return True
        else:
            Hybrid_Agents_Battery_Limit_str = self.raw_list[4].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Battery_Limit_list = Hybrid_Agents_Battery_Limit_str.split(',')
            if not Hybrid_Agents_Battery_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Battery_Limit_str)== 0:
                    self.adv_setting_list[4] = self.default_list[4]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agent battery limit must be digits')
                    return False
            elif len(Hybrid_Agents_Battery_Limit_list)!= (self.robo_team_para[2]):
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agent battery limit must be the same as the robot team parameter')
                return False
            else:
                battery_flag = digit_validate(Hybrid_Agents_Battery_Limit_list, 200, 800, self.robo_team_para[2])
                if not battery_flag:
                    QMessageBox.warning(self, 'Warning', 'Hybrid agents battery limit must between 200 and 800')
                    return False
                else:
                    self.adv_setting_list[4] = list_division_single(Hybrid_Agents_Battery_Limit_list, self.robo_team_para[2])
                    return True

    def Perception_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[0]) == 0:
            self.adv_setting_list[5] = []
            return True
        else:
            Perception_Agents_Velocity_Limit_str = self.raw_list[5].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Perception_Agents_Velocity_Limit_list = Perception_Agents_Velocity_Limit_str.split(',')
            if not Perception_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Perception_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[5] = self.default_list[5]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Perception agent velocity limit must be digits')
                    return False
            elif len(Perception_Agents_Velocity_Limit_list)!= (self.robo_team_para[0]):
                QMessageBox.warning(self, 'Warning', 'The number of the perception agent velocity limit must be the same as the robot team parameter')
                return False
            else:
                velocity_flag = digit_validate(Perception_Agents_Velocity_Limit_list, 10, 30, self.robo_team_para[0])
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Perception agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[5] = list_division_single(Perception_Agents_Velocity_Limit_list, self.robo_team_para[0])
                    return True

    def Action_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[6] = []
            return True
        else:
            Action_Agents_Velocity_Limit_str = self.raw_list[6].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Velocity_Limit_list = Action_Agents_Velocity_Limit_str.split(',')
            if not Action_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[6] = self.default_list[6]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Action agent velocity limit must be digits')
                    return False
            elif len(Action_Agents_Velocity_Limit_list)!= (self.robo_team_para[1]):
                QMessageBox.warning(self, 'Warning', 'The number of the action agent velocity limit must be the same as the robot team parameter')
                return False
            else:
                velocity_flag = digit_validate(Action_Agents_Velocity_Limit_list, 10, 30, self.robo_team_para[1])
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[6] = list_division_single(Action_Agents_Velocity_Limit_list, self.robo_team_para[1])
                    return True

    def Hybrid_Agents_Velocity_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[7] = []
            return True
        else:
            Hybrid_Agents_Velocity_Limit_str = self.raw_list[7].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Velocity_Limit_list = Hybrid_Agents_Velocity_Limit_str.split(',')
            if not Hybrid_Agents_Velocity_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Velocity_Limit_str)== 0:
                    self.adv_setting_list[7] = self.default_list[7]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agent velocity limit must be digits')
                    return False
            elif len(Hybrid_Agents_Velocity_Limit_list)!= (self.robo_team_para[2]):
                    QMessageBox.warning(self, 'Warning', 'The number of the hybrid agent velocity limit must be the same as the robot team parameter')
                    return False
            else:
                velocity_flag = digit_validate(Hybrid_Agents_Velocity_Limit_list, 10, 30, self.robo_team_para[2])
                if not velocity_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents velocity limit must between 10 and 30')
                    return False
                else:
                    self.adv_setting_list[7] = list_division_single(Hybrid_Agents_Velocity_Limit_list, self.robo_team_para[2])
                    return True

    def Action_Agents_Tank_Capacity_Limit_Determination(self):
        if (self.robo_team_para[1]) == 0:
            self.adv_setting_list[8] = []
            return True
        else:
            Action_Agents_Tank_Capacity_Limit_str = self.raw_list[8].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Action_Agents_Tank_Capacity_Limit_list = Action_Agents_Tank_Capacity_Limit_str.split(',')
            if not Action_Agents_Tank_Capacity_Limit_str.replace(',', '').isdigit():
                if len(Action_Agents_Tank_Capacity_Limit_str)== 0:
                    self.adv_setting_list[8] = self.default_list[8]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Action agent water tank capacity limit must be digits')
                    return False
            elif len(Action_Agents_Tank_Capacity_Limit_list)!= (self.robo_team_para[1]):
                QMessageBox.warning(self, 'Warning', 'The number of the action agent water tank capacity limit must be the same as the robot team parameter')
                return False
            else:
                water_capacity_flag = digit_validate(Action_Agents_Tank_Capacity_Limit_list, 1, 15,
                                                     self.robo_team_para[1])
                if not water_capacity_flag:
                    QMessageBox.warning(self, 'Warning', 'Action agents water tank capacity limit must between 1 and 15')
                    return False
                else:
                    self.adv_setting_list[8] = list_division_single(Action_Agents_Tank_Capacity_Limit_list, self.robo_team_para[1])
                    return True

    def Hybrid_Agents_Tank_Capacity_Limit_Determination(self):
        if (self.robo_team_para[2]) == 0:
            self.adv_setting_list[9] = []
            return True
        else:
            Hybrid_Agents_Tank_Capacity_Limit_str = self.raw_list[9].replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            Hybrid_Agents_Tank_Capacity_Limit_list = Hybrid_Agents_Tank_Capacity_Limit_str.split(',')
            if not Hybrid_Agents_Tank_Capacity_Limit_str.replace(',', '').isdigit():
                if len(Hybrid_Agents_Tank_Capacity_Limit_str)== 0:
                    self.adv_setting_list[9] = self.default_list[9]
                    return True
                else:
                    QMessageBox.warning(self, 'Warning', 'Invalid input: Hybrid agent tank capacity must be digits')
                    return False
            elif len(Hybrid_Agents_Tank_Capacity_Limit_list)!= (self.robo_team_para[2]):
                QMessageBox.warning(self, 'Warning', 'The number of the hybrid agent tank capacity must be the same as the robot team parameter')
                return False
            else:
                water_capacity_flag = digit_validate(Hybrid_Agents_Tank_Capacity_Limit_list, 1, 15, self.robo_team_para[2])
                if not water_capacity_flag:
                    QMessageBox.warning(self, 'Warning', 'Hybrid agents water tank capacity limit must between 1 and 15')
                    return False
                else:
                    self.adv_setting_list[9] = list_division_single(Hybrid_Agents_Tank_Capacity_Limit_list, self.robo_team_para[2])
                    return True

    def set_default(self):
        default_list_Batttery1 = []
        default_list_Velocity1 = []
        height_list = []
        for i in range(self.robo_team_para[0]):
            default_list_Batttery1.append([500])
            default_list_Velocity1.append([20])
            height_list.append([10, 100])

        default_list_Batttery2 = []
        default_list_Velocity2 = []
        default_list_Tank2 = []
        for i in range(self.robo_team_para[1]):
            default_list_Batttery2.append([500])
            default_list_Velocity2.append([20])
            default_list_Tank2.append([10])

        default_list_Batttery3 = []
        default_list_Velocity3 = []
        default_list_Tank3 = []
        height_list1 = []
        for i in range(self.robo_team_para[2]):
            default_list_Batttery3.append([500])
            default_list_Velocity3.append([20])
            default_list_Tank3.append([10])
            height_list1.append([10, 100])

        self.default_list = [height_list, height_list1, default_list_Batttery1, default_list_Batttery2, default_list_Batttery3, default_list_Velocity1, default_list_Velocity2, default_list_Velocity3, default_list_Tank2, default_list_Tank3]


    def string_to_array(self):
        default = False
        if self.perception_agent_altitude_Determination():
            if self.hybrid_agent_altitude_Determination():
                if self.Perception_Agents_Battery_Limit_Determination():
                    if self.Action_Agents_Battery_Limit_Determination():
                        if self.Hybrid_Agents_Battery_Limit_Determination():
                            if self.Perception_Agents_Velocity_Limit_Determination():
                                if self.Action_Agents_Velocity_Limit_Determination():
                                    if self.Hybrid_Agents_Velocity_Limit_Determination():
                                        if self.Action_Agents_Tank_Capacity_Limit_Determination():
                                            if self.Hybrid_Agents_Tank_Capacity_Limit_Determination():
                                                default = True
        return default

    def array_to_string(self, list):
        out_str = '['
        for i in range(len(list)):
            if len(list[i]) == 1:
                out_str += str(list[i][0])
            else:
                out_str += '(' + str(list[i][0]) + ', ' + str(list[i][1]) + ')'
            if i < (len(list) - 1):
                out_str += ', '
        out_str += ']'
        return out_str

    def transmit_value(self):
        for i in range(len(self.raw_list)):
            if len(self.adv_setting_list[i]) == 0:
                self.raw_list[i] = ''
            else:
                self.raw_list[i] = self.array_to_string(self.adv_setting_list[i])

    def back_function(self):
        self.hide()
        self.screen = lake_location_define(self.environment_para, self.robo_team_para, self.set_loci, self.current_grid, self.applied_flag)
        self.screen.show()

    def transfer_function(self):
        self.robo_team_para[3] = 0
        self.adv_setting_list = [[], [], [], [], [], [], [], [], [], []]
        self.hide()
        self.screen = homo_adv_setting(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, self.current_grid, self.applied_flag)
        self.screen.show()

    def skip_function(self):
        self.adv_setting_list = self.default_list
        self.screen = pre_View(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list)
        answer = QMessageBox.question(self, 'Environment Pre-View', "Do you like to proceed with this environment design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.hide()
            self.screen = tutorial(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, 0)
            self.screen.show()

    def next_function(self):
        if self.string_to_array():
            self.screen = pre_View(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list)
            answer = QMessageBox.question(self, 'Environment Pre-View', "Do you like to proceed with this environment design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer == QMessageBox.Yes:
                self.hide()
                self.screen = tutorial(self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting_list, 0)
                self.screen.show()

# Load the utilities for the simulation
Agent_Util = HeteroFireBots_Reconn_Env_Utilities()

# The preview page
class pre_View():
    def __init__(self, environment_para, robo_team_para, set_loci, adv_setting):
        super(pre_View, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_loci
        self.adv_setting = adv_setting
        self.close_flag = 0
        self.plot_scene()

    def agent_arrange(self):
        if self.set_loci[2][1] == 1:
            relative_pos = [[-40, -180], [-40, -120], [-40, -60], [-40, 0], [-40, 60], [-40, 120], [40, -180], [40, -120], [40, -60]]
        else:
            relative_pos = [[-180, -40], [-100, -40], [-20, -40], [60, -40], [140, -40], [-180, 40], [-100, 40], [-20, 40], [60, 40]]
        self.agent_pos_list = []
        for i in range(self.robo_team_para[0] + self.robo_team_para[1] + self.robo_team_para[2]):
            self.agent_pos_list.append([self.set_loci[2][0][0] + relative_pos[i][0], self.set_loci[2][0][1] + relative_pos[i][1]])

    def plot_scene(self):
        world_Size = self.environment_para[0]
        fire_center_Loci = []
        agent_Base_Loci = []
        target_Loci = []
        lake_Loci = []

        for i in range(len(self.set_loci[0])):
            if self.set_loci[1][9] == 0:
                fire_center_Loci.append([self.set_loci[0][i][0], self.set_loci[0][i][1], self.set_loci[1][0]])
            else:
                fire_center_Loci.append([self.set_loci[0][i][0], self.set_loci[0][i][1], self.set_loci[1][0][i]])

        if self.set_loci[2][1] == 1:
            agent_Base_Loci = [[[self.set_loci[2][0][0], self.set_loci[2][0][1], 160, 400, 9, 1, 0]]]
        else:
            agent_Base_Loci = [[[self.set_loci[2][0][0], self.set_loci[2][0][1], 400, 160, 9, 1, 0]]]

        for i in range(len(self.set_loci[3])):
            target_Loci.append([[self.set_loci[3][i][0], self.set_loci[3][i][1], 120, 150, 0, 1, 0]])

        for i in range(len(self.set_loci[4])):
            target_Loci.append([[self.set_loci[4][i][0], self.set_loci[4][i][1], 150, 180, 1, 1, 0]])

        for i in range(len(self.set_loci[5])):
            target_Loci.append([[self.set_loci[5][i][0], self.set_loci[5][i][1], 180, 220, 2, 1, 0]])

        for i in range(len(self.set_loci[6])):
            lake_Loci.append([self.set_loci[6][i][0], self.set_loci[6][i][1]])

        self.agent_arrange()

        pygame.font.init()
        hospital_Font = pygame.font.SysFont('arial', 40)

        # The agent is represented as a blue dot with radius of 8
        agent_Radius = 8
        # The field of view (FOV) of the agent is [pi/4, pi/6], which corresponds to the size of the searching scope
        agent_FOV = [np.pi / 4, np.pi / 6]
        # Set the current font for the text (Agent / Goal)
        font_Agent = pygame.font.SysFont('arial', 30)

        init_height = 20

        screen = pygame.display.set_mode((world_Size, world_Size), 0, 32)
        while True:
            screen.fill((197, 225, 165))
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.close_flag = 1

            Agent_Util.road_plot(screen, target_Loci)
            Agent_Util.agent_Base_Plot(screen, 1, agent_Base_Loci, 0)

            Agent_Util.fire_region_plot(screen, hospital_Font, fire_center_Loci)

            Agent_Util.lake_plot_static(screen, lake_Loci)

            for i in range(len(self.set_loci[3]) + len(self.set_loci[4]) + len(self.set_loci[5])):
                Agent_Util.target_Plot(screen, hospital_Font, target_Loci[i], 0)

            # Go over all the elements in the current_Agent_State_List
            for i in range(self.robo_team_para[0]):
                # Plot all the searching agents
                # Calculate the size of the searching scope
                searching_Scope_X = 2 * np.tan(agent_FOV[0]) * init_height
                searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * init_height

                # The coordination of the upper-left corner of the agent searching scope
                searching_Scope_Upper_Left_Corner = (self.agent_pos_list[i][0] - searching_Scope_X / 2,
                                                     self.agent_pos_list[i][1] - searching_Scope_Y / 2)
                # The size of the agent searching scope
                searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                # Plot the searching agent (Circle) and its corresponding searching scope (Rectangle)
                pygame.draw.circle(screen, (0, 0, 255), (int(self.agent_pos_list[i][0]), int(self.agent_pos_list[i][1])),
                                   agent_Radius)
                pygame.draw.rect(screen, (0, 0, 255), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

                # Display the name of each agent
                screen.blit(font_Agent.render('P' + str(i + 1), False, (0, 0, 0)), (self.agent_pos_list[i][0], self.agent_pos_list[i][1]))

            for i in range(self.robo_team_para[1]):
                # Plot all the firefighter agents
                # The vertex set for the firefighter agent
                firefighter_Agent_Vertex = [(self.agent_pos_list[i + self.robo_team_para[0]][0] - agent_Radius, self.agent_pos_list[i + self.robo_team_para[0]][1]),
                                            (self.agent_pos_list[i + self.robo_team_para[0]][0], self.agent_pos_list[i + self.robo_team_para[0]][1] + agent_Radius),
                                            (self.agent_pos_list[i + self.robo_team_para[0]][0] + agent_Radius, self.agent_pos_list[i + self.robo_team_para[0]][1]),
                                            (self.agent_pos_list[i + self.robo_team_para[0]][0], self.agent_pos_list[i + self.robo_team_para[0]][1]- agent_Radius)]
                # Plot the firefighter agent (Diamond)
                pygame.draw.polygon(screen, (128, 0, 128), firefighter_Agent_Vertex)

                # Display the name of each agent
                screen.blit(font_Agent.render('A' + str(i + 1), False, (0, 0, 0)), (
                self.agent_pos_list[i + self.robo_team_para[0]][0], self.agent_pos_list[i + self.robo_team_para[0]][1]))

            for i in range(self.robo_team_para[2]):
                # Plot all the hybrid agents
                # Calculate the size of the searching scope
                searching_Scope_X = 2 * np.tan(agent_FOV[0]) * init_height
                searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * init_height

                # The coordination of the upper-left corner of the agent searching scope
                searching_Scope_Upper_Left_Corner = (self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0] - searching_Scope_X / 2,
                                                     self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1] - searching_Scope_Y / 2)
                # The size of the agent searching scope
                searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                # The vertex set for the hybrid agent
                firefighter_Agent_Vertex = [(self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0] - agent_Radius, self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1]),
                                            (self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0], self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1] + agent_Radius),
                                            (self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0] + agent_Radius, self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1]),
                                            (self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0], self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1] - agent_Radius)]
                # Plot the hybrid agent (Diamond) and its corresponding searching scope (Rectangle)
                pygame.draw.polygon(screen, (0, 128, 128), firefighter_Agent_Vertex)

                pygame.draw.rect(screen, (0, 128, 128), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

                # Display the name of each agent
                screen.blit(font_Agent.render('H' + str(i + 1), False, (0, 0, 0)), (self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][0], self.agent_pos_list[i + self.robo_team_para[0] + self.robo_team_para[1]][1]))


            if self.close_flag == 0:
                pygame.display.update()
            else:
                pygame.quit()
                break

# Essential parameters in the simulation environment
facility_penalty = [1, 2, 5, 5]
score_list = [0, 0, 0, 0, 0, 0]

# The main game simulation class
class environment_ctl():
    def __init__(self, environment_para, robo_team_para, set_loci, adv_setting, scenario_idx):
        super(environment_ctl, self).__init__()
        self.environment_para = environment_para
        self.robo_team_para = robo_team_para
        self.set_loci = set_loci
        self.adv_setting = adv_setting
        self.scenario_idx = scenario_idx
        self.close_flag = 0
        self.environment_simulation()

    def agent_arrange(self):
        if self.set_loci[2][1] == 1:
            relative_pos = [[-40, -180], [-40, -120], [-40, -60], [-40, 0], [-40, 60], [-40, 120], [40, -180], [40, -120], [40, -60]]
        else:
            relative_pos = [[-180, -40], [-100, -40], [-20, -40], [60, -40], [140, -40], [-180, 40], [-100, 40], [-20, 40], [60, 40]]
        self.agent_pos_list = []
        for i in range(self.robo_team_para[0] + self.robo_team_para[1] + self.robo_team_para[2]):
            self.agent_pos_list.append([self.set_loci[2][0][0] + relative_pos[i][0], self.set_loci[2][0][1] + relative_pos[i][1]])

    def environment_simulation(self):
        # ************************ Part 0: Background ******************************
        # All the size unit of the object are represented in pixel
        # Background: (Color: Dark Green (0, 180, 0))
        # The size of the window (simulation environment) in pixel: 1024 * 1024
        world_Size = self.environment_para[0]
        # Display size: The width of the state bar
        display_Size = 600

        score_size_x = 800
        score_size_y = 400

        global facility_penalty
        global score_list
        global username

        # ************************ Part 1: Target ******************************
        # Targets (Color: Orange (255, 165, 0))
        # The number of the targets: 3
        house_Num = self.environment_para[3]
        hospital_Num = self.environment_para[4]
        power_station_Num =self.environment_para[5]

        target_Num = house_Num + hospital_Num + power_station_Num

        # Target info: target center position, target size, target type (0: Normal, 1: Hospital, 2: Power Station)
        # The list to store the position of each target in the dictionary
        target_Loci = []
        target_onFire_list = [[], [], [], []]
        target_onFire_Flag = [[], [], [], []]
        target_info = [[], [], [], []]
        for i in range(house_Num):
            # The center of the target 1: (700, 700), size 120 * 150, Normal, enable_Edge_Flag, current time
            target_Loci.append([[self.set_loci[3][i][0], self.set_loci[3][i][1], 120, 150, 0, 1, 0]])
            target_info[0].append([self.set_loci[3][i][0], self.set_loci[3][i][1], 120, 150, 0, i])
            target_onFire_list[0].append([0])
            target_onFire_Flag[0].append(0)

        for i in range(hospital_Num):
            # The center of the target 2: (600, 800), size 120 * 150, Hospital, current time
            target_Loci.append([[self.set_loci[4][i][0], self.set_loci[4][i][1], 150, 180, 1, 1, 0]])
            target_info[1].append([self.set_loci[4][i][0], self.set_loci[4][i][1], 150, 180, 1, i])
            target_onFire_list[1].append([0])
            target_onFire_Flag[1].append(0)

        for i in range(power_station_Num):
            # The center of the target 2: (600, 800), size 120 * 150, Hospital, current time
            target_Loci.append([[self.set_loci[5][i][0], self.set_loci[5][i][1], 180, 220, 2, 1, 0]])
            target_info[2].append([self.set_loci[5][i][0], self.set_loci[5][i][1], 180, 220, 2, i])
            target_onFire_list[2].append([0])
            target_onFire_Flag[2].append(0)

        lake_list = []
        lake_Loci = []
        for i in range(len(self.set_loci[6])):
            lake_list.append([[self.set_loci[6][i][0], self.set_loci[6][i][1]]])
            lake_Loci.append([[self.set_loci[6][i][0], self.set_loci[6][i][1], 100, 0]])

        # ************************ Part 2: Fire state ******************************
        # Fire region (Color: Red (255, 0, 0))
        # The wildfire generation and propagation utilizes the FARSITE wildfire mathematical model (Thanks to Esi Seraj)
        # To clarify the fire state data, the state of the fire spot at each moment is stored in the dictionary list separately
        # Besides, the current fire map will also be stored as the matrix with the same size of the simulation model, which
        # reflects the fire intensity of each position on the world
        # The number of the fire spots
        fireSpots_Num = self.environment_para[2]
        # Create the fire state dictionary list
        fire_States_List = []
        for i in range(fireSpots_Num):
            fire_States_List.append([])

        terrain_sizes = [world_Size, world_Size]  # length and width of the terrain as a list [length [m], width [m]]
        # [[x_min [m], x_max [m], y_min [m], y_max [m]]]
        hotspot_areas = []
        for i in range(self.environment_para[2]):
            hotspot_areas.append([self.set_loci[0][i][0] - 50, self.set_loci[0][i][0] + 50, self.set_loci[0][i][1] - 50, self.set_loci[0][i][1] + 50])
        duration = self.environment_para[1]  # total duration to run the simulation
        time_point = 0  # current time
        time_step = 0.01  # (sampling frequency)^-1
        fire_decay_flg = False  # declare if dynamic fire decay over time should be included or not
        decay_rate = 0.0005  # fuel exhaustion rate (greater means faster exhaustion)

        if self.set_loci[1][9] == 0:
            num_ign_points = self.set_loci[1][0]  # initial number of fire spots (ignition points) per hotspot area

            fuel_coeff = self.set_loci[1][2]  # fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
            wind_speed = self.set_loci[1][3]  # average mid-flame wind velocity (higher values streches the fire more)
            wind_direction = np.pi * 2 * self.set_loci[1][4] / 360 # wind azimuth

            time_vector = time_point * np.ones(num_ign_points * len(hotspot_areas))  # a vector containing the time passed after ignition of each point
            fire_env = WildFire(
                terrain_sizes=terrain_sizes, hotspot_areas=hotspot_areas, num_ign_points=num_ign_points, duration=duration, time_step=1,
                radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi/3)  # local form

            ign_points_all = fire_env.hotspot_init()  # initializing hotspots
            previous_terrain_map = ign_points_all.copy()  # initializing the starting terrain map
            geo_phys_info = fire_env.geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed, avg_wind_direction=wind_direction)

            fire_map = ign_points_all  # initializing fire-map
        else:
            fire_env = []
            geo_phys_info = []
            ign_points_all = []
            previous_terrain_map = []
            time_vector = []
            new_fire_front_temp = []
            current_geo_phys_info = []
            for i in range(self.environment_para[2]):
                new_fire_front_temp.append([])
                current_geo_phys_info.append([])
                num_ign_points = self.set_loci[1][0][i]  # initial number of fire spots (ignition points) per hotspot area

                fuel_coeff = self.set_loci[1][2][i]  # fuel coefficient for vegetation type of the terrain (higher fuel_coeff:: more circular shape fire)
                wind_speed = self.set_loci[1][3][i]  # average mid-flame wind velocity (higher values streches the fire more)
                wind_direction = np.pi * 2 * self.set_loci[1][4][i] / 360  # wind azimuth

                time_vector.append(time_point * np.ones(num_ign_points * 1))  # a vector containing the time passed after ignition of each point
                fire_env.append(WildFire(
                    terrain_sizes=terrain_sizes, hotspot_areas=[hotspot_areas[i]], num_ign_points=num_ign_points,
                    duration=duration, time_step=1, radiation_radius=10, weak_fire_threshold=5, flame_height=3, flame_angle=np.pi / 3))  # local form
                ign_points_all.append(fire_env[i].hotspot_init())  # initializing hotspots
                previous_terrain_map.append(fire_env[i].hotspot_init())  # initializing the starting terrain map
                geo_phys_info.append(fire_env[i].geo_phys_info_init(max_fuel_coeff=fuel_coeff, avg_wind_speed=wind_speed,
                                                            avg_wind_direction=wind_direction))
            fire_map = []
            for i in range(self.environment_para[2]):
                for j in range(len(ign_points_all[i])):
                    fire_map.append(ign_points_all[i][j])   # initializing fire-map
            fire_map = np.array(fire_map)
            fire_map_spec = ign_points_all

        fire_Current_Map = np.zeros([world_Size,world_Size], dtype=float)
        if self.set_loci[1][9] == 1:
            fire_turnon_flag = np.zeros((self.environment_para[2], 1), dtype=int)
        else:
            fire_turnon_flag = 0

        # The lists to store the firespots in different state, coordinates only
        # The onFire_List, store the points currently on fire (sensed points included, pruned points excluded)
        onFire_List = []
        # The sensed_List, store the points currently on fire and have been sensed by agents
        sensed_List = []
        # The pruned_List, store the pruned fire spots
        pruned_List = []

        fire_Num_list = [0, 0, 0]

        # ************************ Part 3: Agents ******************************
        # Agent base: Rectangle, Yellow(255, 255, 0)
        # The number of the agent's base
        agent_Base_Num = 1

        # The list to store the position of each agent base in the dictionary
        # Agent Base 1: Position (120, 600), Size: (160, 400), Capacity, enable_Edge_Flag, current time
        if self.set_loci[2][1] == 1:
            agent_Base_Loci = [[[self.set_loci[2][0][0], self.set_loci[2][0][1], 160, 400, 9, 1, 0]]]
            target_info[3].append([self.set_loci[2][0][0], self.set_loci[2][0][1], 160, 400, 1, 0])
            target_onFire_list[3].append([0])
            target_onFire_Flag[3].append(0)
        else:
            agent_Base_Loci = [[[self.set_loci[2][0][0], self.set_loci[2][0][1], 400, 160, 9, 1, 0]]]
            target_info[3].append([self.set_loci[2][0][0], self.set_loci[2][0][1], 400, 160, 1, 0])
            target_onFire_list[3].append([0])
            target_onFire_Flag[3].append(0)

        # Agents: Color: Searching agents: Blue (0, 0, 255)
        #                HeteroFireBot_Env agents: Purple (128, 0, 128)
        #                Hybrid agents: Cyan (0, 255, 255)
        # The number of the searching agents
        searching_Agent_Num = self.robo_team_para[0]
        # The number of the firefighter agents
        firefighter_Agent_Num = self.robo_team_para[1]
        # The number of the hybrid agents
        hybrid_Agent_Num = self.robo_team_para[2]

        # The agent is represented as a blue dot with radius of 8
        agent_Radius = 8
        # The field of view (FOV) of the agent is [pi/4, pi/6], which corresponds to the size of the searching scope
        agent_FOV = [np.pi/4, np.pi/6]

        # The upper bound of the agent's flight (Meter)
        agent_Upper_Height_List = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)
        for i in range(searching_Agent_Num):
            if self.robo_team_para[3] == 0:
                agent_Upper_Height_List[i] = self.adv_setting[0][0][1]
            else:
                agent_Upper_Height_List[i] = self.adv_setting[0][i][1]

        for i in range(firefighter_Agent_Num):
            agent_Upper_Height_List[searching_Agent_Num + i] = 30

        for i in range(hybrid_Agent_Num):
            if self.robo_team_para[3] == 0:
                agent_Upper_Height_List[searching_Agent_Num + firefighter_Agent_Num + i] = self.adv_setting[1][0][1]
            else:
                agent_Upper_Height_List[searching_Agent_Num + firefighter_Agent_Num + i] = self.adv_setting[1][i][1]

        # The lower bound of the agent's flight (Meter)
        agent_Lower_Height_List = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)
        for i in range(searching_Agent_Num):
            if self.robo_team_para[3] == 0:
                agent_Lower_Height_List[i] = self.adv_setting[0][0][0]
            else:
                agent_Lower_Height_List[i] = self.adv_setting[0][i][0]

        for i in range(firefighter_Agent_Num):
            agent_Lower_Height_List[searching_Agent_Num + i] = 30

        for i in range(hybrid_Agent_Num):
            if self.robo_team_para[3] == 0:
                agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i] = self.adv_setting[1][0][0]
            else:
                agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i] = self.adv_setting[1][i][0]

        # lift_Step is the length of each height change after pressing the keyboard up and down button (Meter)
        lift_Step = 5

        # Battery parameter [Total energy, consumption during flight, consumption during waiting]
        battery_para = []
        for i in range(searching_Agent_Num):
            if self.robo_team_para[3] == 0:
                battery_para.append([self.adv_setting[2][0][0], 0.1, 0.05])
            else:
                battery_para.append([self.adv_setting[2][i][0], 0.1, 0.05])

        for i in range(firefighter_Agent_Num):
            if self.robo_team_para[3] == 0:
                battery_para.append([self.adv_setting[3][0][0], 0.1, 0.05])
            else:
                battery_para.append([self.adv_setting[3][i][0], 0.1, 0.05])

        for i in range(hybrid_Agent_Num):
            if self.robo_team_para[3] == 0:
                battery_para.append([self.adv_setting[4][0][0], 0.1, 0.05])
            else:
                battery_para.append([self.adv_setting[4][i][0], 0.1, 0.05])

        # Start flag, if the start flag is 0, silent
        # If the start flag is 1, the agent begins to move
        # If the start flag is 2, return
        start_Flag = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)

        # Move mode flag, if its value is 0, the agent will automatically fly towards the goal
        # If its value is 1, the agent will fly by one step when clicking
        move_Mode_Flag = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)

        self.agent_arrange()

        # Initialize the list to store the agent current state at each moment
        current_Agent_State_List = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            current_Agent_State_List.append([])

        # Initialize the list to store the current state of each agent
        # Content: [agent_Position(X, Y, Z), agent_Velocity(X, Y, Z), goal_Index, current_Time, agent' type, agent index,
        #          sum of running distance, sum of waiting time, water tank capacity, move_enabling_flag, patrolling flag,
        #          patrolling goal]
        for i in range(searching_Agent_Num):
            current_Agent_State_List[i] = [self.agent_pos_list[i][0], self.agent_pos_list[i][1], max(20, agent_Lower_Height_List[i]), 0, 0, 0, 0, 0, 0, i + 1, 0, 0, 0, 1, 0, 0]

        for i in range(firefighter_Agent_Num):
            if self.robo_team_para[3] == 0:
                current_Agent_State_List[searching_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + i][0], self.agent_pos_list[searching_Agent_Num + i][1],
                                                                    30, 0, 0, 0, 0, 0, 1, i + 1, 0, 0, self.adv_setting[8][0][0], 1, 0, 0]
            else:
                current_Agent_State_List[searching_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + i][0], self.agent_pos_list[searching_Agent_Num + i][1],
                                                                     30, 0, 0, 0, 0, 0, 1, i + 1, 0, 0, self.adv_setting[8][i][0], 1, 0, 0]

        for i in range(hybrid_Agent_Num):
            if self.robo_team_para[3] == 0:
                current_Agent_State_List[searching_Agent_Num + firefighter_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][0],
                                                                                            self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][1],
                                                                                            max(20, agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i]),
                                                                                            0, 0, 0, 0, 0, 2, i + 1, 0, 0, self.adv_setting[9][0][0], 1, 0, 0]
            else:
                current_Agent_State_List[searching_Agent_Num + firefighter_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][0],
                                                                                             self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][1],
                                                                                             max(20, agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i]),
                                                                                             0, 0, 0, 0, 0, 2, i + 1, 0, 0, self.adv_setting[9][i][0], 1, 0, 0]
        # Initial height of the agents
        agent_Init_Pos_Z = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            agent_Init_Pos_Z.append(current_Agent_State_List[i][2])

        # agent_Current_Pos_Z is the current flight height of the agent
        agent_Current_Pos_Z = agent_Init_Pos_Z

        # Initialize the original agent state list
        original_Agent_State_List = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            original_Agent_State_List.append([])

        # Preserve the original info for each agent
        for i in range(searching_Agent_Num):
            original_Agent_State_List[i] = [self.agent_pos_list[i][0], self.agent_pos_list[i][1], max(20, agent_Lower_Height_List[i]), 0, 0, 0, 0, 0, 0, i + 1, 0, 0, 0, 1, 0, 0]

        for i in range(firefighter_Agent_Num):
            if self.robo_team_para[3] == 0:
                original_Agent_State_List[searching_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + i][0], self.agent_pos_list[searching_Agent_Num + i][1],
                                                                    30, 0, 0, 0, 0, 0, 1, i + 1, 0, 0, self.adv_setting[8][0][0], 1, 0, 0]
            else:
                original_Agent_State_List[searching_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + i][0], self.agent_pos_list[searching_Agent_Num + i][1],
                                                                     30, 0, 0, 0, 0, 0, 1, i + 1, 0, 0, self.adv_setting[8][i][0], 1, 0, 0]

        for i in range(hybrid_Agent_Num):
            if self.robo_team_para[3] == 0:
                original_Agent_State_List[searching_Agent_Num + firefighter_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][0],
                                                                                            self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][1],
                                                                                            max(20, agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i]),
                                                                                            0, 0, 0, 0, 0, 2, i + 1, 0, 0, self.adv_setting[9][0][0], 1, 0, 0]
            else:
                original_Agent_State_List[searching_Agent_Num + firefighter_Agent_Num + i] = [self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][0],
                                                                                             self.agent_pos_list[searching_Agent_Num + firefighter_Agent_Num + i][1],
                                                                                             max(20, agent_Lower_Height_List[searching_Agent_Num + firefighter_Agent_Num + i]),
                                                                                             0, 0, 0, 0, 0, 2, i + 1, 0, 0, self.adv_setting[9][i][0], 1, 0, 0]

        # Initialize the list to store the selected agent current state at each moment
        current_Agent_State = []

        # current agent index is the current index of the agent in operation
        # For the current program, 0, 1 are the searching agent, 2, 3 are the firefighter agent
        current_Agent_Index = 0

        # Th initial maximum fire intensity
        current_Max_Intensity = 155

        # Initialize the global list to store the agent state at each moment
        global_Agent_State = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            global_Agent_State.append([])

        # Initialize the patrolling goal list to store the current patrolling goal for each agent
        patrolling_Goal_List = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            patrolling_Goal_List.append([])

        # Initialize the goal index list to store the index for each goal for reference
        goal_Index_List = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            goal_Index_List.append([])

        # Initialize the list to store the sensed fire spot data
        sensed_Fire_Spot_List = []
        for i in range(searching_Agent_Num + hybrid_Agent_Num):
            sensed_Fire_Spot_List.append([])

        # Initialize the list to store the pruned fire spot data
        pruned_Fire_Spot_List = []
        for i in range(firefighter_Agent_Num + hybrid_Agent_Num):
            pruned_Fire_Spot_List.append([])

        # Initialize the list to store the waiting time for the action agent only
        waiting_Time_List = []
        for i in range(firefighter_Agent_Num):
            waiting_Time_List.append(0)

        # The upper bound of the agent's flight (Meter)
        confidence_level_list = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=float)
        for i in range(searching_Agent_Num):
            confidence_level_list[i] = 0

        for i in range(firefighter_Agent_Num):
            confidence_level_list[searching_Agent_Num + i] = self.set_loci[1][7]/100

        for i in range(hybrid_Agent_Num):
            confidence_level_list[searching_Agent_Num + firefighter_Agent_Num + i] = self.set_loci[1][8]/100

        # Initialize the trigger for pruning the fire spot
        pruning_Trigger = np.zeros(firefighter_Agent_Num + hybrid_Agent_Num)

        # Initialize the list to store the CoM info of sensed fire spot
        CoM_Info_List = []
        for i in range(searching_Agent_Num + hybrid_Agent_Num):
            CoM_Info_List.append([])

        # ************************ Part 4: User Data ******************************
        # goal_Index: store the index of the goal (Action type 0, Mouse click)
        goal_Index = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)
        # current_Goal_Index: the position of the current goal in user_data_list
        current_Goal_Index = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num, dtype=int)

        # Initialize the list to store the history of the agent's goal
        # Content: The goal position (X, Y), current time, action type, goal_index
        # Action type: -1: Keyboard Down, 0: Mouse click, 1: Keyboard Up
        # goal_index will only be utilized when action 0 happens. For the rest of the actions, it will remain 0
        # Initial the global_User_Data_List to store all the goals of each agent
        global_User_Data_List = []
        for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
            global_User_Data_List.append([])

        # Initialize the index for each agent's next goal
        index_Next = np.zeros(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num)

        # Initialize the pygame environment
        pygame.init()

        # Initialize the global clock
        # The simulation time is counted in seconds, while the actual time is counted in milliseconds
        clock = pygame.time.Clock()

        # Initialize the buffer that stores the last storage time
        last_store_time = 0

        # Create a screen (Width * Height) = (1024 * 1024)
        screen = pygame.display.set_mode((world_Size + display_Size, world_Size), 0, 32)

        # Set the title of the window
        pygame.display.set_caption("HeteroFireBots_Reconn_Env Agent Simulation Environment")

        # Initialize the text display
        pygame.font.init()

        # Set the current font for the text (Agent / Goal)
        font_Agent = pygame.font.SysFont('arial', 30)
        # Set the current font for the side bar
        font_Side = pygame.font.SysFont('arial', 20)

        font_Side_Title = pygame.font.SysFont('arial', 24)
        #font_Side_Title.set_bold(True)

        font_Side_Bold = pygame.font.SysFont('arial', 24)
        #font_Side_Bold.set_bold(True)

        font_Score = pygame.font.SysFont('arial', 28)

        font_Scorelist = pygame.font.SysFont('arial', 26)

        # Set the current font for the hospital word
        hospital_Font = pygame.font.SysFont('arial', 40)

        # Video recording flag, simulation video will only be made when this flag equals to 1
        video_Recording_Flag = 0

        # ************************ Background Information Storage ******************************
        # Create the list to store the background information
        background_Info = [self.environment_para, self.robo_team_para, self.set_loci, self.adv_setting, agent_Radius, agent_FOV, 16]
        # Write the background information into the .pkl file
        if self.scenario_idx == 0:
            background_Info_Output = open('Dependencies/Open_World_Data/' + username + '/Background_Info.pkl', 'wb')
            pickle.dump(background_Info, background_Info_Output)

            # Write the information of the agent's battery into the .pkl file
            battery_Info_Output = open('Dependencies/Open_World_Data/' + username + '/Battery_Info.pkl', 'wb')
            pickle.dump(battery_para, battery_Info_Output)
        elif self.scenario_idx > 0:
            background_Info_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Background_Info.pkl', 'wb')
            pickle.dump(background_Info, background_Info_Output)

            # Write the information of the agent's battery into the .pkl file
            battery_Info_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Battery_Info.pkl', 'wb')
            pickle.dump(battery_para, battery_Info_Output)

        # ************************ Main loop for display ******************************
        while True:
            # Fill the screen with green
            screen.fill((197, 225, 165))

            # Acquire the current time (Synchronize the time stamp)
            current_Time = pygame.time.get_ticks()
            # Extract the info and goal set of the current selected agent from the whole list
            current_Agent_State = current_Agent_State_List[current_Agent_Index]
            agent_Upper_Height = agent_Upper_Height_List[current_Agent_Index]
            agent_Lower_Height = agent_Lower_Height_List[current_Agent_Index]

            # Assume the agent moves in the constant speed, the maximum speed is 20
            if current_Agent_State[8] == 0:
                if self.robo_team_para[3] == 0:
                    agent_Speed = self.adv_setting[5][0][0]
                else:
                    agent_Speed = self.adv_setting[5][current_Agent_State[9] - 1][0]
            elif current_Agent_State[8] == 1:
                if self.robo_team_para[3] == 0:
                    agent_Speed = self.adv_setting[6][0][0]
                else:
                    agent_Speed = self.adv_setting[6][current_Agent_State[9] - 1][0]
            else:
                if self.robo_team_para[3] == 0:
                    agent_Speed = self.adv_setting[7][0][0]
                else:
                    agent_Speed = self.adv_setting[7][current_Agent_State[9] - 1][0]

            # ************************ Part 1: Monitor the click event ******************************
            # Set the exit event to generate the simulation video and exit the simulation environment
            for event in pygame.event.get():
                if event.type == QUIT:
                    if video_Recording_Flag == 1:
                        # Generate the video to record the simulation
                        if self.scenario_idx == 0:
                            Util.generate_animation('Dependencies/Open_World_Data/' + username + '/Raw_Images',
                                                    'Dependencies/Open_World_Data/' + username + '/Open_World' + '_' + username + '_Animation.avi')
                        elif self.scenario_idx > 0:
                            Util.generate_animation('Dependencies/Scenario_Data/Scenario#' + str(
                                self.scenario_idx) + "/" + username + '/Raw_Images',
                                                    'Dependencies/Scenario_Data/Scenario#' + str(
                                                        self.scenario_idx) + "/" + username + '/Scenario#' + str(
                                                        self.scenario_idx) + '_' + username + '_Animation.avi')

                    if self.scenario_idx == 0:
                        # Remove the directory 'Raw_Images'
                        shutil.rmtree('Dependencies/Open_World_Data/' + username + '/Raw_Images')
                        # Re-create the directory 'Raw_Images'
                        os.makedirs('Dependencies/Open_World_Data/' + username + '/Raw_Images')
                    elif self.scenario_idx > 0:
                        # Remove the directory 'Raw_Images'
                        shutil.rmtree('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')
                        # Re-create the directory 'Raw_Images'
                        os.makedirs('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')

                    # Exit the simulation environment
                    self.close_flag = 1
                    #sys.exit()

                # If the left button of the mouse is pressed, get the current position of the mouse (Goal)
                elif event.type == MOUSEBUTTONDOWN:
                    # Get the position of the mouse when the click event happens
                    (goal_X, goal_Y) = pygame.mouse.get_pos()

                    # Ensure that the mouse click happens inside the window
                    if (goal_X <= world_Size) and (goal_Y <= world_Size):
                        in_Base_Flag, base_Index = Agent_Util.in_Agent_Base_Region(goal_X, goal_Y, agent_Base_Num, agent_Base_Loci)
                        # If the clicked position locates within the base region, the agent will go back the base to refuel
                        if in_Base_Flag:
                            current_Agent_State[13] = 2

                        else:
                            # Update the goal_Index variable
                            goal_Index[current_Agent_Index] += 1

                            # Create the buffer to store the location and time(ms) of the new goal (Action 0)
                            new_Goal_Buffer = [goal_X, goal_Y, np.floor(current_Time / 100), 0, goal_Index[current_Agent_Index]]

                            if (start_Flag[current_Agent_Index] == 0) and (current_Agent_State[7] > 0):
                                current_Goal_Index[current_Agent_Index] = max(len(goal_Index_List[current_Agent_Index]),
                                                                              current_Goal_Index[current_Agent_Index])
                                patrolling_Goal_List[current_Agent_Index] = [new_Goal_Buffer]
                                current_Agent_State[6] = max(len(goal_Index_List[current_Agent_Index]) + 1, current_Agent_State[6])
                                current_Agent_State[14] = 0
                                current_Agent_State[15] = 0

                            # If patrolling agent flag is 0, add it into the patrolling goal list
                            if (current_Agent_State[14] == 0):
                                # Set the first goal as the first element in the patrolling goal list
                                if (len(global_User_Data_List[current_Agent_Index]) == 0):
                                    patrolling_Goal_List[current_Agent_Index].append(new_Goal_Buffer)
                                else:
                                    # Determine the closure of the patrolling loop
                                    patrolling_Distance = np.sqrt(
                                        (new_Goal_Buffer[0] - patrolling_Goal_List[current_Agent_Index][0][0]) ** 2 +
                                        (new_Goal_Buffer[1] - patrolling_Goal_List[current_Agent_Index][0][1]) ** 2)
                                    # If the distance between the new goal and the 1st element in the patrolling loop list is less than the agent_Speed,
                                    # set the flag of the patrolling loop closure as 1
                                    if (patrolling_Distance <= (agent_Speed * 5)) and (len(patrolling_Goal_List[current_Agent_Index]) > 1):
                                        current_Agent_State[14] = 1
                                        index = 0
                                        for i in range(len(patrolling_Goal_List[current_Agent_Index])):
                                            if current_Agent_State[6] == patrolling_Goal_List[current_Agent_Index][i][4]:
                                                index = i
                                                break
                                        current_Agent_State[15] = index

                                    # If not, add it into the patrolling goal list
                                    else:
                                        patrolling_Goal_List[current_Agent_Index].append(new_Goal_Buffer)
                            # If patrolling agent flag is 1, determine the ending condition of the patrolling loop
                            else:
                                current_Goal_Index[current_Agent_Index] = max(len(goal_Index_List[current_Agent_Index]), current_Goal_Index[current_Agent_Index])
                                current_Agent_State[6] = max(len(goal_Index_List[current_Agent_Index]) + 1, current_Agent_State[6])
                                current_Agent_State[14] = 0
                                current_Agent_State[15] = 0
                                patrolling_Goal_List[current_Agent_Index] = [new_Goal_Buffer]

                            # Record the index of the current goal in the global_User_Data_List
                            goal_Index_List[current_Agent_Index].append(len(global_User_Data_List[current_Agent_Index]))

                            # Record the goal information in the global list
                            global_User_Data_List[current_Agent_Index].append(new_Goal_Buffer)

                            if self.scenario_idx == 0:
                                # Write the information of the goals into the .pkl file
                                user_Data_Output = open('Dependencies/Open_World_Data/' + username + '/User_Data.pkl', 'wb')
                                pickle.dump(global_User_Data_List, user_Data_Output)
                            elif self.scenario_idx > 0:
                                # Write the information of the goals into the .pkl file
                                user_Data_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/User_Data.pkl', 'wb')
                                pickle.dump(global_User_Data_List, user_Data_Output)

                            # Update the start flag
                            if current_Agent_State[13] != 2:
                                start_Flag[current_Agent_Index] = 1

                        # If its value of the move mode flag is 1, the agent will fly by one step when clicking
                        if move_Mode_Flag[current_Agent_Index] == 1:
                            current_Agent_State, agent_Current_Pos_Z[current_Agent_Index], agent_Init_Pos_Z[current_Agent_Index], \
                            current_Goal_Index[current_Agent_Index], pruning_Trigger, start_Flag[current_Agent_Index], waiting_Time_List = \
                                Agent_Util.agent_Motion_Controller(agent_Speed, current_Agent_State,
                                                                 move_Mode_Flag[current_Agent_Index],
                                                                 goal_X, goal_Y, agent_Current_Pos_Z[current_Agent_Index],
                                                                 agent_Init_Pos_Z[current_Agent_Index],
                                                                 global_User_Data_List[current_Agent_Index],
                                                                 current_Goal_Index[current_Agent_Index],
                                                                 goal_Index[current_Agent_Index],
                                                                 start_Flag[current_Agent_Index], firefighter_Agent_Num,
                                                                 pruning_Trigger, battery_para[current_Agent_Index],
                                                                 original_Agent_State_List[current_Agent_Index],
                                                                 patrolling_Goal_List[current_Agent_Index], waiting_Time_List, current_Time)

                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State

                # If the up or down button of the keyboard is pressed, change the size of the searching scope
                elif event.type == KEYDOWN:
                    # Mark the keyboard action for storage
                    keyboard_Action_Type = 0

                    # If the up button of the keyboard is pressed, enlarge the searching scope
                    if event.key == pygame.K_UP:
                        # This action is only effective for the sensing agents and hybrid agents
                        if (current_Agent_State[8] == 0) or (current_Agent_State[8] == 2):
                            # Update the flight height
                            agent_Current_Pos_Z[current_Agent_Index] = agent_Init_Pos_Z[current_Agent_Index] + lift_Step
                            # Ensure the height value does not exceed the upper bound
                            if agent_Current_Pos_Z[current_Agent_Index] > agent_Upper_Height:
                                agent_Current_Pos_Z[current_Agent_Index] = agent_Upper_Height
                            # Set the keyboard action as 1
                            keyboard_Action_Type = 1

                    # If the down button of the keyboard is pressed, shrink the searching scope
                    elif event.key == pygame.K_DOWN:
                        # This action is only effective for the sensing agents and hybrid agents
                        if (current_Agent_State[8] == 0) or (current_Agent_State[8] == 2):
                            # Update the flight height
                            agent_Current_Pos_Z[current_Agent_Index] = agent_Init_Pos_Z[current_Agent_Index] - lift_Step
                            # Ensure the height value does not exceed the lower bound
                            if agent_Current_Pos_Z[current_Agent_Index] < agent_Lower_Height:
                                agent_Current_Pos_Z[current_Agent_Index] = agent_Lower_Height
                            # Set the keyboard action as -1
                            keyboard_Action_Type = -1

                    # If digit key between 1 and 9 on the keyboard is pressed, switch the agent index
                    elif event.key == pygame.K_1:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 1:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 0

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_2:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 2:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 1

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_3:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 3:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 2

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_4:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 4:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 3

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_5:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 5:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 4

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_6:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 6:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 5

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_7:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 7:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 6

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_8:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 8:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 7

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    elif event.key == pygame.K_9:
                        if (searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num) >= 9:
                            # Update the current_Agent_State_List first
                            # Update the corresponding element in the current_Agent_State_List
                            current_Agent_State_List[current_Agent_Index] = current_Agent_State
                            # Append to the current agent state list into the corresponding list
                            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                                global_Agent_State[i] += current_Agent_State_List[i]

                            # Switch the agent index
                            current_Agent_Index = 8

                            # Update the current_Agent_State list
                            current_Agent_State = current_Agent_State_List[current_Agent_Index]

                    # If the keyboard up or down event happens, write the action to the list
                    if keyboard_Action_Type != 0:
                        # Create the buffer to store the location and time(ms) of the new goal (Action -1 / 1)
                        new_Goal_Buffer = [current_Agent_State[0], current_Agent_State[1],
                                           np.floor(current_Time/100), keyboard_Action_Type, 0]
                        # Record the goal information in the global list
                        global_User_Data_List[current_Agent_Index].append(new_Goal_Buffer)

                        if self.scenario_idx == 0:
                            # Write the information of the goals into the .pkl file
                            user_Data_Output = open('Dependencies/Open_World_Data/' + username + '/User_Data.pkl', 'wb')
                            pickle.dump(global_User_Data_List, user_Data_Output)
                        elif self.scenario_idx > 0:
                            # Write the information of the goals into the .pkl file
                            user_Data_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/User_Data.pkl', 'wb')
                            pickle.dump(global_User_Data_List, user_Data_Output)


            # ************************ Part 2: Plot the targets ******************************
            Agent_Util.road_plot(screen, target_Loci)
            # Plot all the targets
            for i in range(target_Num):
                target_Loci[i] = Agent_Util.target_Plot(screen, hospital_Font, target_Loci[i], current_Time)

            lake_Loci = Agent_Util.lake_plot(screen, lake_list, lake_Loci, current_Time)

            # ************************ Part 3: Plot the fire spot ******************************
            # propagate the wildfire
            if self.set_loci[1][9] == 0:
                if current_Time > (self.set_loci[1][1] * 1000):
                    new_fire_front, current_geo_phys_info = fire_env.fire_propagation(world_Size, ign_points_all=ign_points_all,
                        geo_phys_info=geo_phys_info, previous_terrain_map=previous_terrain_map, pruned_List = pruned_List)
                    updated_terrain_map = previous_terrain_map
                else:
                    new_fire_front = np.array([])
                    current_geo_phys_info = []
            else:
                updated_terrain_map = previous_terrain_map
                for i in range(self.environment_para[2]):
                    if current_Time > (self.set_loci[1][1][i] * 1000):
                        new_fire_front_temp[i], current_geo_phys_info[i] = fire_env[i].fire_propagation(world_Size,
                                                                                          ign_points_all=ign_points_all[i],
                                                                                          geo_phys_info=geo_phys_info[i],
                                                                                          previous_terrain_map=previous_terrain_map[i],
                                                                                          pruned_List=pruned_List)
                    else:
                        new_fire_front_temp[i] = np.array([])
                        current_geo_phys_info[i] = []
                new_fire_front = []
                for i in range(self.environment_para[2]):
                    for j in range(len(new_fire_front_temp[i])):
                        new_fire_front.append(new_fire_front_temp[i][j])
                new_fire_front = np.array(new_fire_front)

            if self.set_loci[1][9] == 1:
                for i in range(self.environment_para[2]):
                    if current_Time > (self.set_loci[1][1][i] * 1000):
                        fire_map_spec[i] = np.concatenate([fire_map_spec[i], new_fire_front_temp[i]], axis=0)
                    else:
                        fire_map_spec[i] = fire_map_spec[i]
            else:
                fire_map_spec = fire_map

            # Process the fire spot information
            fire_Current_Map, fire_States_List, onFire_List, target_onFire_list, target_onFire_Flag = Agent_Util.fire_Data_Storage(self.set_loci[1][0], fire_States_List,
                                    new_fire_front, world_Size, fireSpots_Num, fire_Current_Map, current_Time, onFire_List, target_onFire_list, target_onFire_Flag, target_info, self.set_loci[1][9], fire_turnon_flag)

            # ************************ Part 4: Update the position of the agent ******************************
            agent_Base_Loci = Agent_Util.agent_Base_Plot(screen, agent_Base_Num, agent_Base_Loci, current_Time)

            # Search all the existing agents, update its position
            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                # If its value of the move mode flag is 0, the agent will automatically fly towards the goal
                if ((move_Mode_Flag[i] == 0) and (len(global_User_Data_List[i]) > 0)):
                    # Current selected agent
                    if i == current_Agent_Index:
                        current_Agent_State, agent_Current_Pos_Z[i], agent_Init_Pos_Z[i], \
                        current_Goal_Index[i], pruning_Trigger, start_Flag[i], waiting_Time_List\
                          = Agent_Util.agent_Motion_Controller(agent_Speed, current_Agent_State, move_Mode_Flag[i], 0.0, 0.0,
                                                             agent_Current_Pos_Z[i], agent_Init_Pos_Z[i],
                                                             global_User_Data_List[i], current_Goal_Index[i], goal_Index[i],
                                                             start_Flag[i], firefighter_Agent_Num, pruning_Trigger,
                                                             battery_para[i], original_Agent_State_List[i],
                                                             patrolling_Goal_List[i], waiting_Time_List, current_Time)
                        # Update the corresponding element in the current_Agent_State_List
                        current_Agent_State_List[i] = current_Agent_State

                    else:
                        # Rest agent, moving towards to the previous goal automatically
                        current_Agent_State1, agent_Current_Pos_Z[i], agent_Init_Pos_Z[i], \
                        current_Goal_Index[i], pruning_Trigger, start_Flag[i], waiting_Time_List \
                          = Agent_Util.agent_Motion_Controller(agent_Speed, current_Agent_State_List[i], move_Mode_Flag[i],
                                                             0.0, 0.0, agent_Current_Pos_Z[i], agent_Init_Pos_Z[i],
                                                             global_User_Data_List[i], current_Goal_Index[i], goal_Index[i],
                                                             start_Flag[i], firefighter_Agent_Num, pruning_Trigger,
                                                             battery_para[i], original_Agent_State_List[i],
                                                             patrolling_Goal_List[i], waiting_Time_List, current_Time)
                        # Update the corresponding element in the current_Agent_State_List
                        current_Agent_State_List[i] = current_Agent_State1

                # Append to the current agent state list into the corresponding list
                global_Agent_State[i] = global_Agent_State[i] + current_Agent_State_List[i]

            # ************************ Part 5: Sensing the fire spots ******************************
                # If the current agent is the sensing agent, enable the sensing function
                if (current_Agent_State_List[i][8] == 0):
                    fire_Sensed_Map, CoM_Info, sensed_List = Agent_Util.fire_Sensing(fire_map_spec, current_Agent_State_List[i],
                                                agent_FOV, geo_phys_info, onFire_List, sensed_List, world_Size, self.set_loci[1][0],
                                                self.set_loci[1][9], fire_turnon_flag, [agent_Lower_Height_List[i], agent_Upper_Height_List[i], current_Agent_State_List[i][2]])

                    sensed_Fire_Spot_List[current_Agent_State_List[i][9] - 1].append(fire_Sensed_Map)
                    CoM_Info_List[current_Agent_State_List[i][9] - 1].append(CoM_Info)

                # For the firefighter agents, utilize the pruning function and update the onFire, sensed, and pruned list
                elif (current_Agent_State_List[i][8] == 1):
                    trigger_Index = current_Agent_State_List[i][9] - 1 + firefighter_Agent_Num * (current_Agent_State_List[i][8] - 1)
                    if ((pruning_Trigger[trigger_Index] == 1) and (current_Agent_State_List[i][2] == 30)):
                        fire_Pruned_Map, fire_map, onFire_List, sensed_List, pruned_List, new_fire_front, target_onFire_list, sensed_flag = \
                            Agent_Util.fire_Pruning(fire_map, current_Agent_State_List[i], agent_FOV, onFire_List, sensed_List,
                                                  pruned_List, new_fire_front, target_onFire_list, target_info, confidence_level_list[i])
                        if sensed_flag == 1:
                            pruned_Fire_Spot_List[trigger_Index].append\
                                ([fire_Pruned_Map, [current_Agent_State_List[i][0], current_Agent_State_List[i][1], current_Time]])


                            if current_Agent_State_List[i][12] > 0:
                                current_Agent_State_List[i][12] = current_Agent_State_List[i][12] - 1
                            else:
                               current_Agent_State_List[i][13] = 0

                            pruning_Trigger[trigger_Index] = 2
                    else:
                        pruned_Fire_Spot_List[trigger_Index].append([])

                # For the hybrid agents, enable both function of the sensing and firefighter agents
                elif (current_Agent_State_List[i][8] == 2):
                    fire_Sensed_Map, CoM_Info, sensed_List = Agent_Util.fire_Sensing(fire_map_spec, current_Agent_State_List[i],
                                                 agent_FOV, geo_phys_info, onFire_List, sensed_List, world_Size, self.set_loci[1][0],
                                                 self.set_loci[1][9], fire_turnon_flag, [agent_Lower_Height_List[i], agent_Upper_Height_List[i], current_Agent_State_List[i][2]])

                    sensed_Fire_Spot_List[searching_Agent_Num + current_Agent_State_List[i][9] - 1].append(fire_Sensed_Map)
                    CoM_Info_List[searching_Agent_Num + current_Agent_State_List[i][9] - 1].append(CoM_Info)

                    trigger_Index = current_Agent_State_List[i][9] - 1 + firefighter_Agent_Num * (current_Agent_State_List[i][8] - 1)
                    if ((pruning_Trigger[trigger_Index] == 1) and (current_Agent_State_List[i][2] == 20)):
                        fire_Pruned_Map, fire_map, onFire_List, sensed_List, pruned_List, new_fire_front, target_onFire_list, sensed_flag = \
                            Agent_Util.fire_Pruning(fire_map, current_Agent_State_List[i], agent_FOV, onFire_List, sensed_List,
                                                  pruned_List, new_fire_front, target_onFire_list, target_info, confidence_level_list[i])
                        if sensed_flag == 1:
                            pruned_Fire_Spot_List[trigger_Index].append \
                                ([fire_Pruned_Map, [current_Agent_State_List[i][0], current_Agent_State_List[i][1], current_Time]])

                            if current_Agent_State_List[i][12] > 0:
                                current_Agent_State_List[i][12] = current_Agent_State_List[i][12] - 1

                            else:
                               current_Agent_State_List[i][13] = 0

                            pruning_Trigger[trigger_Index] = 2
                    else:
                        pruned_Fire_Spot_List[trigger_Index].append([])

            # Plot the sensed fire spot for method learning
            current_Max_Intensity = Agent_Util.sensed_Fire_Spot_Plot(screen, sensed_List, fire_Current_Map,
                                                                   current_Max_Intensity)

            # Plot the pruned fire dots
            for i in range(len(pruned_List)):
                # Ensure that all the fire spots to be displayed must be within the window scope
                if ((pruned_List[i][0] <= world_Size) and (pruned_List[i][1] <= world_Size)
                        and (pruned_List[i][0] >= 0) and (pruned_List[i][1] >= 0)):
                    pygame.draw.circle(screen, (0, 0, 0), (pruned_List[i][0], pruned_List[i][1]), 1)

            # updating the fire-map data for next step
            if new_fire_front.shape[0] > 0:
                fire_map = np.concatenate([fire_map, new_fire_front], axis=0)  # raw fire map without fire decay

            if self.set_loci[1][9] == 1:
                ign_points_all_temp = []
                for i in range(self.environment_para[2]):
                    if new_fire_front_temp[i].shape[0] > 0:
                        previous_terrain_map[i] = np.concatenate((updated_terrain_map[i], new_fire_front_temp[i]), axis=0)  # fire map with fire decay

                    if current_Time > self.set_loci[1][1][i] * 1000:
                        fire_turnon_flag[i] = 1
                        ign_points_all_temp.append(new_fire_front_temp[i])
                    else:
                        ign_points_all_temp.append(ign_points_all[i])
                ign_points_all = ign_points_all_temp
            else:
                if current_Time > self.set_loci[1][1] * 1000:
                    fire_turnon_flag = 1
                if new_fire_front.shape[0] > 0:
                    previous_terrain_map = np.concatenate((updated_terrain_map, new_fire_front))  # fire map with fire decay
                    ign_points_all = new_fire_front

            # ************************ Part 6: Plot all the agents ******************************
            # Initialize the text display buffer
            text = []

            # Go over all the elements in the current_Agent_State_List
            for i in range(searching_Agent_Num + firefighter_Agent_Num + hybrid_Agent_Num):
                # Plot all the searching agents
                if current_Agent_State_List[i][8] == 0:
                    # Calculate the size of the searching scope
                    searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State_List[i][2]
                    searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State_List[i][2]

                    # The coordination of the upper-left corner of the agent searching scope
                    searching_Scope_Upper_Left_Corner = (current_Agent_State_List[i][0] - searching_Scope_X / 2,
                                                         current_Agent_State_List[i][1] - searching_Scope_Y / 2)
                    # The size of the agent searching scope
                    searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                    # Plot the searching agent (Circle) and its corresponding searching scope (Rectangle)
                    pygame.draw.circle(screen, (0, 0, 255), (int(current_Agent_State_List[i][0]), int(current_Agent_State_List[i][1])),
                                       agent_Radius)
                    pygame.draw.rect(screen, (0, 0, 255), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('P' + str(current_Agent_State_List[i][9]), False, (0, 0, 0)))

                    # Display the firefighter agent's trajectory
                    index_Next[i] = Agent_Util.traj_Plot(screen, global_User_Data_List[i], current_Agent_State_List[i],
                                                       patrolling_Goal_List[i], (0, 0, 255))

                # Plot all the firefighter agents
                elif current_Agent_State_List[i][8] == 1:
                    # The vertex set for the firefighter agent
                    firefighter_Agent_Vertex = [(current_Agent_State_List[i][0] - agent_Radius, current_Agent_State_List[i][1]),
                                                (current_Agent_State_List[i][0], current_Agent_State_List[i][1] + agent_Radius),
                                                (current_Agent_State_List[i][0] + agent_Radius, current_Agent_State_List[i][1]),
                                                (current_Agent_State_List[i][0], current_Agent_State_List[i][1]- agent_Radius)]
                    # Plot the firefighter agent (Diamond)
                    pygame.draw.polygon(screen, (128, 0, 128), firefighter_Agent_Vertex)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('A' + str(current_Agent_State_List[i][9]), False, (0, 0, 0)))

                    # Display the firefighter agent's trajectory
                    index_Next[i] = Agent_Util.traj_Plot(screen, global_User_Data_List[i], current_Agent_State_List[i],
                                                       patrolling_Goal_List[i], (128, 0, 128))

                # Plot all the hybrid agents
                elif current_Agent_State_List[i][8] == 2:
                    # Calculate the size of the searching scope
                    searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State_List[i][2]
                    searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State_List[i][2]

                    # The coordination of the upper-left corner of the agent searching scope
                    searching_Scope_Upper_Left_Corner = (current_Agent_State_List[i][0] - searching_Scope_X / 2,
                                                         current_Agent_State_List[i][1] - searching_Scope_Y / 2)
                    # The size of the agent searching scope
                    searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                    # The vertex set for the hybrid agent
                    firefighter_Agent_Vertex = [(current_Agent_State_List[i][0] - agent_Radius, current_Agent_State_List[i][1]),
                                                (current_Agent_State_List[i][0], current_Agent_State_List[i][1] + agent_Radius),
                                                (current_Agent_State_List[i][0] + agent_Radius, current_Agent_State_List[i][1]),
                                                (current_Agent_State_List[i][0], current_Agent_State_List[i][1] - agent_Radius)]
                    # Plot the hybrid agent (Diamond) and its corresponding searching scope (Rectangle)
                    pygame.draw.polygon(screen, (0, 128, 128), firefighter_Agent_Vertex)

                    pygame.draw.rect(screen, (0, 128, 128), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('H' + str(current_Agent_State_List[i][9]), False, (0, 0, 0)))

                    # Display the hybrid agent's trajectory
                    index_Next[i] = Agent_Util.traj_Plot(screen, global_User_Data_List[i], current_Agent_State_List[i],
                                                       patrolling_Goal_List[i], (0, 128, 128))

                # Display the name of each agent
                screen.blit(text[i], (current_Agent_State_List[i][0], current_Agent_State_List[i][1]))

                # Display the goal list of the corresponding agent
                Agent_Util.goal_Marker(screen, font_Agent, global_User_Data_List[i], current_Agent_State_List[i],
                                     patrolling_Goal_List[i], move_Mode_Flag[i])

            # ************************ Part 7: Plot the center of mass ******************************

            # ************************ Part 8: Plot the display bar ******************************
            # Display the side bar
            pygame.draw.rect(screen, (211, 211, 211), Rect((world_Size, 0), (display_Size, world_Size)))

            # Update the state info
            pos = Agent_Util.side_Bar_Display(screen, current_Agent_State_List, font_Side, font_Side_Bold, font_Side_Title, battery_para,
                                      global_User_Data_List, index_Next, goal_Index_List, world_Size)

            # ************************ Part 9: Compute the game score ******************************
            overall_pruning_score, preception_score, action_score, safe_Num, facility_perception_score, total_Negative_Score, total_Negative_percent = \
                Agent_Util.score_Calculation(len(fire_map), onFire_List, sensed_List, pruned_List, target_onFire_list, target_onFire_Flag, facility_penalty, self.environment_para, self.set_loci, current_Time)

            score_list = [overall_pruning_score, preception_score, action_score, safe_Num, facility_perception_score, total_Negative_Score, total_Negative_percent]
            Agent_Util.score_display(screen, font_Side_Bold, font_Score, font_Scorelist, pos, score_list)

            # ************************ Part 10: Save the .pkl and the images ******************************
            # Save the current state on the screen
            # # All the data files will be saved every 200ms (Frequency = 5 Hz)
            if video_Recording_Flag == 1:
                if (pygame.time.get_ticks() % 100) == 0:
                    if self.scenario_idx == 0:
                        pygame.image.save(screen, 'Dependencies/Open_World_Data/' + username + '/Raw_Images/' + str(current_Time) + ".png")
                    elif self.scenario_idx > 0:
                        pygame.image.save(screen, 'Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images/' + str(current_Time) + ".png")

            if self.scenario_idx == 0:
                # All the data files will be saved every 200ms (Frequency = 5 Hz)
                if (pygame.time.get_ticks() - last_store_time) >= 200:
                    # Write the information of the targets into the .pkl file
                    target_Loci_Output = open('Dependencies/Open_World_Data/' + username + '/Target_Loci.pkl', 'wb')
                    pickle.dump(target_Loci, target_Loci_Output)

                    # Write the information of the fire states into the .pkl file
                    fire_States_Output = open('Dependencies/Open_World_Data/' + username + '/Fire_States.pkl', 'wb')
                    pickle.dump(fire_States_List, fire_States_Output)

                    # Write the current fire spot map into the .pkl file
                    fire_Map_Output = open('Dependencies/Open_World_Data/' + username + '/Fire_Map.pkl', 'wb')
                    pickle.dump(fire_Current_Map, fire_Map_Output)

                    # Write the lake info into the .pkl file
                    lake_Output = open('Dependencies/Open_World_Data/' + username + '/Lake_info.pkl', 'wb')
                    pickle.dump(lake_Loci, lake_Output)

                    # Write the center of mass information into the .pkl file
                    CoM_Output = open('Dependencies/Open_World_Data/' + username + '/Sensing_Data_CoM.pkl', 'wb')
                    pickle.dump(CoM_Info_List, CoM_Output)

                    # Write the current sensed fire spot info into the .pkl file
                    sensed_Fire_Map_Output = open('Dependencies/Open_World_Data/' + username + '/Sensed_Fire_Map.pkl', 'wb')
                    pickle.dump(sensed_Fire_Spot_List, sensed_Fire_Map_Output)

                    # Write the list of the fire fronts that locates inside the targets info into the .pkl file
                    target_onFire_List_Output = open('Dependencies/Open_World_Data/' + username + '/target_onFire_List.pkl', 'wb')
                    pickle.dump(target_onFire_list, target_onFire_List_Output)

                    # Write the current pruned fire spot info into the .pkl file
                    pruned_Fire_Map_Output = open('Dependencies/Open_World_Data/' + username + '/Pruned_Fire_Map.pkl', 'wb')
                    pickle.dump(pruned_Fire_Spot_List, pruned_Fire_Map_Output)

                    # Write the information of the agent base into the .pkl file
                    agent_Base_Loci_Output = open('Dependencies/Open_World_Data/' + username + '/Agent_Base_Loci.pkl', 'wb')
                    pickle.dump(agent_Base_Loci, agent_Base_Loci_Output)

                    # Write the information of the global agent state into the .pkl file
                    agent_States_Output = open('Dependencies/Open_World_Data/' + username + '/Agent_States.pkl', 'wb')
                    pickle.dump(global_Agent_State, agent_States_Output)

                    last_store_time = pygame.time.get_ticks()

            elif self.scenario_idx > 0:
                # All the data files will be saved every 200ms (Frequency = 5 Hz)
                if (pygame.time.get_ticks() - last_store_time) >= 200:
                    # Write the information of the targets into the .pkl file
                    target_Loci_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Target_Loci.pkl', 'wb')
                    pickle.dump(target_Loci, target_Loci_Output)

                    # Write the information of the fire states into the .pkl file
                    fire_States_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Fire_States.pkl', 'wb')
                    pickle.dump(fire_States_List, fire_States_Output)

                    # Write the current fire spot map into the .pkl file
                    fire_Map_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Fire_Map.pkl', 'wb')
                    pickle.dump(fire_Current_Map, fire_Map_Output)

                    # Write the lake info into the .pkl file
                    lake_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Lake_info.pkl', 'wb')
                    pickle.dump(lake_Loci, lake_Output)

                    # Write the center of mass information into the .pkl file
                    CoM_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Sensing_Data_CoM.pkl', 'wb')
                    pickle.dump(CoM_Info_List, CoM_Output)

                    # Write the current sensed fire spot info into the .pkl file
                    sensed_Fire_Map_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Sensed_Fire_Map.pkl', 'wb')
                    pickle.dump(sensed_Fire_Spot_List, sensed_Fire_Map_Output)

                    # Write the list of the fire fronts that locates inside the targets info into the .pkl file
                    target_onFire_List_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/target_onFire_List.pkl', 'wb')
                    pickle.dump(target_onFire_list, target_onFire_List_Output)

                    # Write the current pruned fire spot info into the .pkl file
                    pruned_Fire_Map_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Pruned_Fire_Map.pkl', 'wb')
                    pickle.dump(pruned_Fire_Spot_List, pruned_Fire_Map_Output)

                    # Write the information of the agent base into the .pkl file
                    agent_Base_Loci_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Agent_Base_Loci.pkl', 'wb')
                    pickle.dump(agent_Base_Loci, agent_Base_Loci_Output)

                    # Write the information of the global agent state into the .pkl file
                    agent_States_Output = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Agent_States.pkl', 'wb')
                    pickle.dump(global_Agent_State, agent_States_Output)

                    last_store_time = pygame.time.get_ticks()

            if (current_Time//1000) >= self.environment_para[1]:
                self.close_flag = 1

            # Update the display according to the latest change
            if self.close_flag == 0:
                pygame.display.update()
            else:
                pygame.quit()
                global simulated_flag
                simulated_flag = 1
                break

# The score computation page
class game_over(QWidget):
    def __init__(self, scenario_idx):
        super(game_over, self).__init__()
        global score_list
        self.score_list = score_list
        self.scenario_idx = scenario_idx
        self.init_ui()

    def init_ui(self):
        self.resize(1600, 1000)
        background_Img = QPalette()
        background_Img.setBrush(QPalette.Background, QBrush(QPixmap("./Dependencies/Images/background.png")))
        self.setPalette(background_Img)
        self.setWindowTitle('Game Over')

        self.right_upper_x = 400
        self.right_upper_y = 140

        self.left_y_interval = 40
        self.text_display_len = 600
        self.text_height = 40

        font = QFont('arial')
        font.setPointSize(22)

        font_button = QFont('arial')
        font_button.setPointSize(24)

        font_button2 = QFont('arial')
        font_button2.setPointSize(20)

        font_Title = QFont('arial', 24, 75)
        font_Bold = QFont('arial', 22, 75)

        self.General_Judge = QLabel(self)
        self.General_Judge.setGeometry(self.right_upper_x, self.right_upper_y, self.text_display_len, self.text_height)
        self.General_Judge.setFont(font_Title)
        self.General_Judge.setObjectName("General_Judge")
        self.General_Judge.setText("General Evaluation:")

        font_pe = QPalette()

        self.General_Judge_Text = QLabel(self)
        self.General_Judge_Text.setGeometry(self.right_upper_x + self.text_display_len + 20, self.right_upper_y, self.text_display_len, self.text_height)
        self.General_Judge_Text.setFont(font_Title)
        self.General_Judge_Text.setObjectName("General_Judge_Text")

        if self.score_list[0] >= 0:
            parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 2.5 * self.text_height),
                                           ('Overall Firefighting Performance Score: ', 'overall_score'), (str(self.score_list[0]) + ' / 100'), QColor(60,179,113))
        else:
            parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 2.5 * self.text_height),
                                           ('Overall Firefighting Performance Score: ', 'overall_score'), (str(self.score_list[0]) + ' / 100'), Qt.red)

        parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 3.5 * self.text_height),
                                       ('Perception Score: ', 'perception_score'), (str(self.score_list[1]) + ' / 100'), QColor(60,179,113))

        parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 4.5 * self.text_height),
                                       ('Action Score: ', 'action_score'), (str(self.score_list[2]) + ' / 100'), QColor(60,179,113))

        parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 5.5 * self.text_height),
                                       ('Facility Protection Score: ', 'facilities_protection_score'), (str(self.score_list[4]) + ' / 100'), QColor(60,179,113))

        parameter_Display_Msg_Extended(self, font, font_Bold, (self.right_upper_x, self.right_upper_y + 6.5 * self.text_height),
                                       ('Total Negative Reward Received: ', 'total_neg_reward'), (str(-self.score_list[6]) + ' / 100'), Qt.red)

        self.General_Judge = QLabel(self)
        self.General_Judge.setGeometry(self.right_upper_x, self.right_upper_y + 7.5 * self.text_height, 700, self.text_height)
        self.General_Judge.setFont(font)
        self.General_Judge.setObjectName("General_Judge")
        self.General_Judge.setText("---------------------------------------------------------------------------------------------------------------")

        final_score = round(self.score_list[0] + self.score_list[1] + self.score_list[2] + self.score_list[4] - 3 * self.score_list[6], 2)
        if final_score >= 0:
            parameter_Display_Msg_Extended(self, font, font_Bold,
                                           (self.right_upper_x, self.right_upper_y + 8.5 * self.text_height),
                                           ('Your Game Score is: ', 'game_score'), str(final_score), QColor(60,179,113))
        else:
            parameter_Display_Msg_Extended(self, font, font_Bold,
                                           (self.right_upper_x, self.right_upper_y + 8.5 * self.text_height),
                                           ('Your Game Score is: ', 'game_score'), str(final_score), Qt.red)

        if final_score > 90:
            self.General_Judge_Text.setText("Excellent")
            font_pe.setColor(QPalette.WindowText, QColor(22, 163, 26))
        elif final_score > 70:
            self.General_Judge_Text.setText("Well Done")
            font_pe.setColor(QPalette.WindowText, QColor(146, 208, 80))
        elif final_score > 60:
            self.General_Judge_Text.setText("Almost There")
            font_pe.setColor(QPalette.WindowText, QColor(83, 155, 212))
        elif final_score > 50:
            self.General_Judge_Text.setText("Fair")
            font_pe.setColor(QPalette.WindowText, QColor(255, 80, 80))
        else:
            self.General_Judge_Text.setText("Failed")
            font_pe.setColor(QPalette.WindowText, QColor(192, 0, 0))

        self.General_Judge_Text.setPalette(font_pe)

        self.menu = QPushButton(self)
        self.menu.setGeometry(125, 770, 300, 75)
        self.menu.setFont(font_button2)
        self.menu.setText('Menu')
        self.menu.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/white_300_75.png)}")
        self.menu.clicked.connect(self.back_function)

        self.scenario_mode = QPushButton(self)
        self.scenario_mode.setGeometry(475, 770, 300, 75)
        self.scenario_mode.setFont(font_button2)
        self.scenario_mode.setText('Scenario Mode')
        self.scenario_mode.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.scenario_mode.clicked.connect(self.scenario_mode_function)

        self.save = QPushButton(self)
        self.save.setGeometry(825, 770, 300, 75)
        self.save.setFont(font_button2)
        self.save.setText('Save Playback')
        self.save.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/yellow_300_75.png)}")
        self.save.clicked.connect(self.skip_function)

        self.exit = QPushButton(self)
        self.exit.setGeometry(1175, 770, 300, 75)
        self.exit.setFont(font_button2)
        self.exit.setText('Exit')
        self.exit.setStyleSheet("QPushButton{border-image: url(./Dependencies/Images/orange_300_75.png)}")
        self.exit.clicked.connect(self.exit_function)

    def paintEvent(self, e):
        painter2 = QPainter(self)
        painter2.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter2.setBrush(QBrush(QColor(197, 225, 165)))
        painter2.drawRect(110, 90, 1400, 800)

        painter3 = QPainter(self)
        painter3.setPen(QPen(QColor(47, 82, 143), 2, Qt.SolidLine))
        painter3.setBrush(QBrush(Qt.white))
        painter3.drawRect(140, 110, 1340, 100)
        painter3.drawRect(140, 220, 1340, 530)

    def back_function(self):
        self.hide()
        self.screen = welcome()
        self.screen.show()

    def scenario_mode_function(self):
        self.hide()
        self.screen = scenario_mode()
        self.screen.show()

    def skip_function(self):
        self.hide()
        self.screen = Animation_Reconstruction(self.scenario_idx)
        sys.exit()

    def exit_function(self):
        global username
        if self.scenario_idx == 0:
            shutil.rmtree('Dependencies/Open_World_Data/' + username + '/Raw_Images')
            os.makedirs('Dependencies/Open_World_Data/' + username + '/Raw_Images')
        elif self.scenario_idx > 0:
            shutil.rmtree('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')
            os.makedirs('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')
        sys.exit()

# Load the utilities for animation reconstruction
Animation_Util = Animation_Reconstruction_Reconn_Utilities()

# The animation reconstruction class
class Animation_Reconstruction():
    def __init__(self, scenario_idx):
        super(Animation_Reconstruction, self).__init__()
        self.scenario_idx = scenario_idx
        self.animation()

    def animation(self):
        global username
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Background_Info.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Background_Info.pkl', 'rb')

        [environment_para, robo_team_para, set_loci, adv_setting, agent_Radius, agent_FOV, state_List_Size] = pickle.load(file_IO)
        world_Size = environment_para[0]
        fireSpots_Num = environment_para[2]
        num_ign_points = set_loci[1][0]

        # Load the target info'
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Target_Loci.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Target_Loci.pkl', 'rb')
        target_Loci = pickle.load(file_IO)

        # Load the agent base info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Agent_Base_Loci.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Agent_Base_Loci.pkl', 'rb')
        agent_Base_Loci = pickle.load(file_IO)

        # Load the agent state info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Agent_States.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Agent_States.pkl', 'rb')
        global_Agent_State_List = pickle.load(file_IO)

        # Load the fire state info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Fire_States.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Fire_States.pkl', 'rb')
        fire_States_List = pickle.load(file_IO)

        # Load the sensed fire info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Sensed_Fire_Map.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Sensed_Fire_Map.pkl', 'rb')
        sensed_Fire_List = pickle.load(file_IO)

        # Load the pruned fire info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Pruned_Fire_Map.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Pruned_Fire_Map.pkl', 'rb')
        pruned_Fire_List = pickle.load(file_IO)

        # Load the lake fire info
        if self.scenario_idx == 0:
            file_IO = open('Dependencies/Open_World_Data/' + username + '/Lake_info.pkl', 'rb')
        elif self.scenario_idx > 0:
            file_IO = open('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Lake_info.pkl', 'rb')
        lake_list = pickle.load(file_IO)
        current_Max_Intensity = 155

        # Initialize the pygame environment
        pygame.init()

        # Set the current font for the text (Agent / Goal)
        font_Agent = pygame.font.SysFont('arial', 30)
        # Set the current font for the side bar
        font_Side = pygame.font.SysFont('arial', 20)
        # Set the current font for the hospital word
        hospital_Font = pygame.font.SysFont('arial', 40)

        onFire_List = []
        pruned_List = []

        # Initialize the simulation timeline
        clock = len(agent_Base_Loci[0]) - 1

        # Create a screen (Width * Height) = (1024 * 1024)
        screen = pygame.display.set_mode((world_Size, world_Size), 0, 32)
        current_Max_Intensity = 155
        # Set the title of the window
        pygame.display.set_caption("Recreated HeteroFireBot Environment")

        sensed_list = []

        # ************************ Main loop for display ******************************
        for time in range(clock):
            # Fill the screen with green
            screen.fill((197, 225, 165))

            Animation_Util.road_plot(screen, target_Loci)

            # Plot all the targets
            for i in range(len(target_Loci)):
                Animation_Util.target_Plot(screen, hospital_Font, target_Loci[i][time])

            # Plot all the agent base
            for i in range(len(agent_Base_Loci)):
                Animation_Util.agent_Base_Plot(screen, agent_Base_Loci[i][time])

            text = []

            new_fire_front, onFire_List = Animation_Util.onFire_List_Recovery(num_ign_points, fire_States_List,
                                                                              world_Size,
                                                                              fireSpots_Num, onFire_List, time,
                                                                              set_loci[1][7], set_loci[1][1])

            pruned_List = Animation_Util.pruned_List_Recovery(pruned_Fire_List, pruned_List, time)
            Animation_Util.lake_plot(screen, lake_list, time)

            for i in range(len(sensed_Fire_List)):
                if len(sensed_Fire_List[i][time]) > 0:
                    for j in range(len(sensed_Fire_List[i][time])):
                        sensed_list.append(sensed_Fire_List[i][time][j])
            current_Max_Intensity = Animation_Util.sensed_Fire_Spot_Plot(screen, sensed_list, time,
                                                                         current_Max_Intensity)

            # Plot the pruned fire dots
            for i in range(len(pruned_List)):
                # Ensure that all the fire spots to be displayed must be within the window scope
                if ((pruned_List[i][0] <= world_Size) and (pruned_List[i][1] <= world_Size)
                        and (pruned_List[i][0] >= 0) and (pruned_List[i][1] >= 0)):
                    pygame.draw.circle(screen, (0, 0, 0), (pruned_List[i][0], pruned_List[i][1]), 1)

            # Go over all the elements in the current_Agent_State_List
            for i in range(len(global_Agent_State_List)):
                current_Agent_State_List = global_Agent_State_List[i][
                                           time * state_List_Size:(time + 1) * state_List_Size]
                # Plot all the searching agents
                if current_Agent_State_List[8] == 0:
                    # Calculate the size of the searching scope
                    searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State_List[2]
                    searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State_List[2]

                    # The coordination of the upper-left corner of the agent searching scope
                    searching_Scope_Upper_Left_Corner = (current_Agent_State_List[0] - searching_Scope_X / 2,
                                                         current_Agent_State_List[1] - searching_Scope_Y / 2)
                    # The size of the agent searching scope
                    searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                    # Plot the searching agent (Circle) and its corresponding searching scope (Rectangle)
                    pygame.draw.circle(screen, (0, 0, 255),
                                       (int(current_Agent_State_List[0]), int(current_Agent_State_List[1])),
                                       agent_Radius)
                    pygame.draw.rect(screen, (0, 0, 255), Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size),
                                     2)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('P' + str(current_Agent_State_List[9]), False, (0, 0, 0)))


                # Plot all the firefighter agents
                elif current_Agent_State_List[8] == 1:
                    # The vertex set for the firefighter agent
                    firefighter_Agent_Vertex = [
                        (current_Agent_State_List[0] - agent_Radius, current_Agent_State_List[1]),
                        (current_Agent_State_List[0], current_Agent_State_List[1] + agent_Radius),
                        (current_Agent_State_List[0] + agent_Radius, current_Agent_State_List[1]),
                        (current_Agent_State_List[0], current_Agent_State_List[1] - agent_Radius)]
                    # Plot the firefighter agent (Diamond)
                    pygame.draw.polygon(screen, (128, 0, 128), firefighter_Agent_Vertex)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('A' + str(current_Agent_State_List[9]), False, (0, 0, 0)))

                # Plot all the hybrid agents
                elif current_Agent_State_List[8] == 2:
                    # Calculate the size of the searching scope
                    searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State_List[2]
                    searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State_List[2]

                    # The coordination of the upper-left corner of the agent searching scope
                    searching_Scope_Upper_Left_Corner = (current_Agent_State_List[0] - searching_Scope_X / 2,
                                                         current_Agent_State_List[1] - searching_Scope_Y / 2)
                    # The size of the agent searching scope
                    searching_Scope_Size = (searching_Scope_X, searching_Scope_Y)

                    # The vertex set for the hybrid agent
                    firefighter_Agent_Vertex = [
                        (current_Agent_State_List[0] - agent_Radius, current_Agent_State_List[1]),
                        (current_Agent_State_List[0], current_Agent_State_List[1] + agent_Radius),
                        (current_Agent_State_List[0] + agent_Radius, current_Agent_State_List[1]),
                        (current_Agent_State_List[0], current_Agent_State_List[1] - agent_Radius)]
                    # Plot the hybrid agent (Diamond) and its corresponding searching scope (Rectangle)
                    pygame.draw.polygon(screen, (0, 128, 128), firefighter_Agent_Vertex)

                    pygame.draw.rect(screen, (0, 128, 128),
                                     Rect(searching_Scope_Upper_Left_Corner, searching_Scope_Size), 2)

                    # Append it into the text display buffer
                    text.append(font_Agent.render('H' + str(current_Agent_State_List[9]), False, (0, 0, 0)))

                # Display the name of each agent
                screen.blit(text[i], (current_Agent_State_List[0], current_Agent_State_List[1]))

            # Write the current screen shot to the directory
            if self.scenario_idx == 0:
                pygame.image.save(screen, 'Dependencies/Open_World_Data/' + username + '/Raw_Images/' + str(time) + ".png")
            elif self.scenario_idx > 0:
                pygame.image.save(screen, 'Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images/' + str(time) + ".png")

            # Update the display according to the latest change
            pygame.display.update()
            pygame.time.wait(10)

        # Generate the video to record the simulation
        if self.scenario_idx == 0:
            Util.generate_animation('Dependencies/Open_World_Data/' + username + '/Raw_Images', 'Dependencies/Open_World_Data/' + username + '/Open_World' + '_' + username + '_Animation.avi')
        elif self.scenario_idx > 0:
            Util.generate_animation('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images', 'Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Scenario#' + str(self.scenario_idx) + '_' + username + '_Animation.avi')

        if self.scenario_idx == 0:
            shutil.rmtree('Dependencies/Open_World_Data/' + username + '/Raw_Images')
            os.makedirs('Dependencies/Open_World_Data/' + username + '/Raw_Images')
        elif self.scenario_idx > 0:
            shutil.rmtree('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')
            os.makedirs('Dependencies/Scenario_Data/Scenario#' + str(self.scenario_idx) + "/" + username + '/Raw_Images')

        pygame.quit()

# The main function
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    main_GUI = QApplication(sys.argv)
    main_Loop = welcome()
    main_Loop.show()
    sys.exit(main_GUI.exec_())

if __name__ == '__main__':
    file_initialize(24)
    main()