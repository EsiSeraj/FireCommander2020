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

import pygame
from pygame.locals import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import os


# Utilities
class Utilities(object):

    # sensing fire through Kalman estimation (sensing and hybrid UAVs only)
    @staticmethod
    def sense(drone_loci=None, fire_spots=None, current_geo_phys_info=None, EKF=None, mdlynamics=None, terrain_sizes=None,
              sensing_abilities=None, mapping_flg=False):
        """
        this function performs the fire-spot sensing through Kalman estimation for sensing and hybrid UAVs

        :param drone_loci: array of drone locations
        :param fire_spots: array of ground truth fire locations
        :param current_geo_phys_info: Geo-physical information of the terrain
        :param EKF: Initialized EKF environment (object)
        :param mdlynamics: initialized model-dynamics for EKF (object)
        :param terrain_sizes: terrain sizes
        :param sensing_abilities: vector of sensing abilities
        :param mapping_flg: mapping flag used in mdlynamics (default:: False)
        :return: state estimates, drones uncertainty, total error and approximated error map
        """

        if drone_loci is None or fire_spots is None or current_geo_phys_info is None or EKF is None or mdlynamics is None or\
                terrain_sizes is None or sensing_abilities is None:
            raise ValueError(">>> Oops! Function 'sens()' needs ALL its input arguments to work, except for the mapping flag (default:: False).")

        # extracting/receiving geo-physical information for estimation and prediction
        R = current_geo_phys_info[:, 0].reshape(fire_spots.shape[0], 1)
        U = current_geo_phys_info[:, 1].reshape(fire_spots.shape[0], 1)
        Theta = current_geo_phys_info[:, 2].reshape(fire_spots.shape[0], 1)

        # Kalman estimation
        drones_uncertainty_all = []
        error_map = np.zeros(shape=terrain_sizes)
        total_uncertainty = np.zeros(shape=[1, drone_loci.shape[0]])
        d_counter = 0
        drones_sense = []
        for d in drone_loci:
            px, py, pz, ptheta = d[0], d[1], d[2], d[3]
            counter = 0
            this_drones_uncertainty_about_q = np.zeros(shape=[fire_spots.shape[0], 1])
            state_estimates_all = []
            for q in fire_spots:
                qx, qy, qint = q[0], q[1], q[2]

                # initializing AEKF parameters
                sensing_coeff = sensing_abilities[d_counter]
                x0, P0, Q0, R0 = mdlynamics.kf_initializer(qx, qy, px, py, pz, R[counter][0], U[counter][0], Theta[counter][0], sensing_coeff)
                state_grads = mdlynamics.state_gradients(R[counter][0], U[counter][0], Theta[counter][0])
                F = mdlynamics.state_jacobian(state_grads)
                H = mdlynamics.observation_jacobian(state_grads, mapping_flg)  # direct state estimation with no mapping

                # initializing the EKF
                ekf = EKF(F=F, H=H, xhat0=x0, P0=P0, Q=Q0, R=R0, num_iter=500)

                # EKF predictor
                state_estimates, uncertainties = ekf.ekf(z=None, measurement_quality=1)
                state_estimates_all.append(state_estimates)

                # propagating uncertainty
                uncertainty_about_q = np.sum(uncertainties)
                kf_fire_spot_estimate = np.array([state_estimates[0], state_estimates[1], 0.])
                kf_drone_pose_estimate = np.array([state_estimates[2], state_estimates[3], state_estimates[4]])
                distance_error = np.linalg.norm(kf_fire_spot_estimate - kf_drone_pose_estimate)
                this_drones_uncertainty_about_q[counter] = uncertainty_about_q * distance_error

                # updating the error-map
                apprx_qx = int(round(state_estimates[0]))
                apprx_qy = int(round(state_estimates[1]))
                apprx_qx = min(apprx_qx, terrain_sizes[0]-1)
                apprx_qy = min(apprx_qy, terrain_sizes[1]-1)
                error_map[apprx_qx, apprx_qy] += this_drones_uncertainty_about_q[counter][0]  # Not really important (just for visualization)
                counter += 1

            drones_sense.append(state_estimates_all)
            total_uncertainty[0, d_counter] = error_map.sum()  # calculating the total uncertainty
            drones_uncertainty_all.append(this_drones_uncertainty_about_q)  # keeping track of all fire-spot uncertainties
            d_counter += 1

        return drones_sense, total_uncertainty, drones_uncertainty_all, error_map

    # fighting the fire (fighting UAVs only)
    @staticmethod
    def fight(drone_loci=None, terrain_map=None, uav_class=None, half_angles=None, fighting_abilities=None, tanker_capacities=None, fire_env=None,
              time_vector=None, pruned_fires=None):
        """
        this function performs the fire fighting through damping the fire intensity, using a scale Heaviside step function.

        :param drone_loci: current UAV locations
        :param uav_class: class of the current UAV
        :param terrain_map: most recent fire map
        :param half_angles: camera half-angles list
        :param fighting_abilities: fighting power coefficients
        :param tanker_capacities: UAV's tanker capacity
        :param fire_env: created fire environment object
        :param time_vector: current time vector of the individual fire spots
        :param pruned_fires: list of peviously pruned fires
        :return: updated fire map after damping the fire, pruned fire spots, updated UAv's tanker capacity and etc.
        """

        if drone_loci is None or terrain_map is None or uav_class is None or half_angles is None or fire_env is None or time_vector is None\
                or pruned_fires is None:
            raise ValueError(">>> Oops! Function 'fight()' needs ALL its input arguments to work.")

        counter = 0
        updated_tanker_capacities = np.zeros(tanker_capacities.shape)
        for d in drone_loci:

            # evaluating the UAV parameters:: half-angle and fighting ability
            fighting_ability_coeff = fighting_abilities[counter]

            if uav_class[counter] == 2:  # class of fixed-wings
                half_angle = half_angles[2]
            elif uav_class[counter] == 1:  # class of multi-rotors
                half_angle = half_angles[1]
            elif uav_class[counter] == 0:  # class of ground robots
                half_angle = half_angles[0]
            elif uav_class[counter] == 3:  # class of hybrid UAVs
                half_angle = half_angles[3]
            else:
                raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")

            # obtaining the respective FOV coordinates
            fov_corners_all, fov_corners_tr_bl = Utilities.fov_corners(drone_loci=d, half_angle=half_angle)
            bl_x, bl_y = fov_corners_tr_bl[1, 0], fov_corners_tr_bl[1, 1]  # bottom left coordinates
            tr_x, tr_y = fov_corners_tr_bl[0, 0], fov_corners_tr_bl[0, 1]  # top right coordinates
            spot_num = 0

            # updating fire intensities
            for point in terrain_map:
                x = point[0]
                y = point[1]
                if Utilities.in_fov(bl_x=bl_x, bl_y=bl_y, tr_x=tr_x, tr_y=tr_y, x=x, y=y):
                    terrain_map[spot_num, 2] = point[2] / fighting_ability_coeff
                spot_num += 1

            # updating the UAVs tanker capacity (Note that no matter any fire is down there or not, extinguisher is dumped [in sensing UAVs we trust])
            updated_tanker_capacities[counter] = tanker_capacities[counter] - Utilities.fov_area(fov_corners_all=fov_corners_all)

            # pruning dead fire spots from the fire map
            updated_terrain_map, updated_time_vector, burnt_out_fires_new = fire_env.pruning_fire_map(
                updated_terrain_map=terrain_map, updated_time_vector=time_vector)
            # update the time vector for next step
            time_vector = updated_time_vector
            terrain_map = updated_terrain_map
            pruned_fires = np.concatenate((pruned_fires, burnt_out_fires_new))

            counter += 1

        return terrain_map, time_vector, pruned_fires, updated_tanker_capacities

    # generating the fighting path based on received sensing info
    @staticmethod
    def fighting_path_generation(passed_info=None, fighting_paths=None, ready_idx=None):
        """
        this function generats a fighting path based on received sensing info, using regression

        :param passed_info: passed sensing info from sensing UAVs
        :param fighting_paths: previously generated fighting paths
        :param ready_idx: active UAV indexes
        :return: a fighting path for fighting fixed-wings (close to a unicycle path)
        """

        if passed_info is None or fighting_paths is None or ready_idx is None:
            raise ValueError(">>> Oops! Sensing information are not passed ['None' received]. Define '[]' if no info is yet available.")

        counter = 0
        used_info_idx = []
        for info in passed_info:
            if info.shape[0] == 0:
                pass
            else:
                for this_info in passed_info:
                    # info_len = this_info.shape[0]  # todo: adaptive resolution???
                    if counter >= len(ready_idx.tolist()):
                        break
                    fighting_paths[ready_idx[counter]].append(Utilities.fit_curve(points=this_info, degree=3, resolution=50))
                    used_info_idx.append(counter)
                    counter += 1

                passed_info = [i for j, i in enumerate(passed_info) if j not in used_info_idx]  # removing used info

        return fighting_paths, passed_info

    # curve fitting
    @staticmethod
    def fit_curve(points=None, degree=3, resolution=50):
        """
        this function performs the curve fitting using 'Least squares polynomial fit' to generate fighting path
        The solution minimizes the squared error

        :param points: the sensed points from sensing UAVs
        :param degree: polynomial degree to be fit [default:: 3 - close to unicycle dynamics]
        :param resolution: how many points on the determined trajectory you need [default:: 50 - CANNOT be too small]
        :return:
        """

        if points is None:
            raise ValueError(""">>> Oops! Points are not passed for curve to be generated.""")

        # get x and y vectors
        x = points[:, 0]
        y = points[:, 1]

        # calculate polynomial
        z = np.polyfit(x, y, degree)  # compute polynomial coefficients
        f = np.poly1d(z)  # compute the polynomial

        # calculate new x's and y's
        x_new = np.linspace(x[0], x[-1], resolution)
        y_new = f(x_new)

        # forming the path list
        aa = np.array([x_new.tolist()])
        bb = np.array([y_new.tolist()])
        curve = np.hstack((aa.T, bb.T))

        return curve

    # activity status update
    @staticmethod
    def status_update(drone_stats=None, uav_type=None, base_locations=None):
        """
        this function updates the status of each UAV

        :param drone_stats: Num_UAVx12 stat vector
        :param uav_type: class of UAV
        :param base_locations: Num_Basex3 location of bases in the form of [x, y, z]
        :return: updated drone_stats and a list of indexes of available bases for service
        """

        if drone_stats is None or uav_type is None or base_locations is None:
            raise ValueError(">>> Oops! Function 'status_update()' CANNOT work with None as input values.")

        counter = 0
        for drone_stat in drone_stats:
            # checking the battery availability
            cond1 = []
            for base in base_locations:
                cond1.append(Utilities.time_to_base(states=drone_stat[0:3], depo=base, max_velocity=drone_stat[8]))
            idx = np.where(np.array(cond1) < drone_stat[7])[0]  # check if there's enough battery left to go back to a base
            if len(idx) == 0:
                drone_stats[counter, 10] = 0
                drone_stats[counter, 11] = 1

            # checking the tanker availability
            if uav_type == 'fighting' or uav_type == 'hybrid':
                cond2 = drone_stat[6]
                if cond2 <= 0:
                    drone_stats[counter, 10] = 0
                    drone_stats[counter, 11] = 1

            counter += 1

        return drone_stats

    # base_assignment by distance
    @staticmethod
    def base_assignment_dist(drone_stats=None, base_locations=None):
        """
        this function performs the base assignment by choosing the closest base

        :param drone_stats: UAVs stats
        :param base_locations: list of base locations
        :return: assigned base locations for each UAV
        """

        if drone_stats is None or base_locations is None:
            raise ValueError(">>> Oops! Function 'base_assignment()' needs ALL of its input arguments to work!")

        counter = 0
        assigned_bases = np.zeros(shape=[drone_stats.shape[0], 3])
        for drone_stat in drone_stats:
            # checking the disstances
            times_all = []
            for base in base_locations:
                times_all.append(Utilities.time_to_base(states=drone_stat[0:3], depo=base, max_velocity=drone_stat[8]))

            assigned_bases[counter] = base_locations[np.argmin(np.array(times_all))]

        return assigned_bases

    # assigning bases by solving a constraint satisfaction problem (CSP)
    @staticmethod
    def base_assignment_csp(drone_stats=None, base_locations=None, base_capacities=None):
        """
        this function performs the base assignment by solving a constraint satisfaction problem (CSP) with UAVs as variables, base locations as
        domains, and UAV-velocity-dependent time to the center of the bases and base capacities as constraints.

        :param drone_stats: UAVs stats
        :param base_locations: list of base locations
        :param base_capacities: list of base capacities
        :return: assigned base locations for each UAV and the updated base capacities after assignment
        """

        if drone_stats is None or base_locations is None or base_capacities is None:
            raise ValueError(">>> Oops! Function 'base_assignment()' needs ALL of its input arguments to work!")

        # initially, all bases with open spot for service are available
        available_bases_idx = np.where(np.array(base_capacities) > 0)[0].tolist()
        available_bases = np.reshape(available_bases_idx * drone_stats.shape[0], (drone_stats.shape[0], len(available_bases_idx))).tolist()

        # available options are then revised based on the time required to get there and battery left
        times_all = []
        options_idx = []
        options_idx_track = []
        for drone_stat in drone_stats:
            # checking the battery availability
            cond1 = []
            for base in base_locations:
                cond1.append(Utilities.time_to_base(states=drone_stat[0:3], depo=base, max_velocity=drone_stat[8]))

            # finding potential bases
            situation_eval = np.where(np.array(cond1) < drone_stat[7])[0]

            if situation_eval.shape[0] != 0:
                options_idx.append(np.where(np.array(cond1) < drone_stat[7])[0])
                options_idx_track.append(np.array(cond1) < drone_stat[7])
            else:
                options_idx.append(np.where((np.array(cond1) / 10) < drone_stat[7])[0])  # critical situation:: pick the closest base
                options_idx_track.append((np.array(cond1) / 10) < drone_stat[7])

            times_all.append(cond1)

        options_len = []
        for options in options_idx:
            options_len.append(len(options))

        assigned_bases = np.zeros(shape=[drone_stats.shape[0], 3])
        for d in range(drone_stats.shape[0]):
            most_critical_idx = np.argmin(options_len)
            most_critical_idx_ops = options_idx[int(most_critical_idx)]
            possible_times = np.array(times_all[d])[most_critical_idx_ops]
            not_available_idx = np.where(options_idx_track[d] == False)[0].tolist()
            actual_index = np.argmin(possible_times) + len(np.where(np.array(not_available_idx) < np.argmin(possible_times))[0])
            if actual_index in available_bases[d]:
                # assigning the base
                assigned_bases[d] = base_locations[np.argmin(possible_times)]

                # updating the new capacities
                base_capacities[np.argmin(possible_times)] -= 1
                available_bases_idx = np.where(np.array(base_capacities) > 0)[0]
                available_bases = np.reshape(available_bases_idx * drone_stats.shape[0], (drone_stats.shape[0], len(available_bases_idx))).tolist()
            else:
                print(">>> No service bases are currently available for this UAV. We'll try again during next itteration.")

            options_len[int(most_critical_idx)] = float('Inf')

        return assigned_bases.tolist(), base_capacities

    # determining active/inactive UAVs
    @staticmethod
    def which_uav(drone_stats=None):
        """
        this function distincs active and inactive UAVs for specific UAV classes

        :param drone_stats: drone stat vector of a class
        :return: active and inactive UAVs of the specified class and their indexes
        """

        if drone_stats is None:
            raise ValueError(">>> Oops! At least one of the drone stat matrices is not specified.")

        active_idx = np.nonzero(drone_stats[:, 10])[0]
        active = drone_stats[active_idx, :]

        inactive_idx = np.where(drone_stats[:, 10] == 0)[0]
        inactive = drone_stats[inactive_idx, :]

        return active, inactive, active_idx, inactive_idx

    # check if conditions are satisfied for agents or not
    @staticmethod
    def at_pose(states=None, poses=None, position_error=0.1, rotation_error=1e2):
        """
        This function checks whether agents are "close enough" to required poses or not

        :param states: 3xNum_agents numpy array of unicycle states
        :param poses: 3xNum_agents numpy array of desired states
        :param position_error: acceptable position error (default:: 0.1)
        :param rotation_error: acceptable position error (default:: 1e2 - large value means we don't care about robot's heading at goal position)
        :return: 1xNum_agents numpy index array of agents that are close enough
        """

        if states is None or poses is None:
            raise ValueError(">>> Oops! Current states and desired poses must be specified.")

        # calculate rotation errors with angle wrapping
        res = states[2, :] - poses[2, :]
        res = np.abs(np.arctan2(np.sin(res), np.cos(res)))

        # calculate position errors
        pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)

        # determine which agents are done
        done = np.nonzero((res <= rotation_error) & (pes <= position_error))

        return done

    # computing the distance between the position of a UAV to its depo
    @staticmethod
    def time_to_base(states=None, depo=None, max_velocity=None):
        """
        this function computs the distance between the position of a UAV to its depo

        :param states: current UAV position
        :param depo: UAV depo location
        :param max_velocity: the maximum velocity by which UAV can fly
        :return: the time required to get to the base [min]
        """

        if states is None or depo is None or max_velocity is None:
            raise ValueError(">>> Oops! Current UAV position, depo location and UAV's maximum velocity must be specified.")

        pos_diff = states - depo
        dist = np.sqrt((pos_diff[0] ** 2) + (pos_diff[1] ** 2) + (pos_diff[2] ** 2))
        time = dist / max_velocity

        return time

    # generating and updating the sensed map ('previous_sensed_map' as opposed to 'previous_terrain_map')
    @staticmethod
    def update_sensed_map(drones_sense=None, intensity_measures=None, sensed_map=None, active_idx=None, time_point=None):
        """
        this function generates the sensed map according to drones sensing results (sensing and hybrid only)

        :param drones_sense: list of drones sense results
        :param intensity_measures: list of fire intensity measures from heat sensors
        :param sensed_map: list of previously sensed map
        :param active_idx: indexes of active UAVs
        :param time_point: current time point
        :return: updated sensed map for all drones
        """

        if drones_sense is None or intensity_measures is None or sensed_map is None or active_idx is None or time_point is None:
            raise ValueError(">>> Oops! drones sense list of previously sensed map and fire intensity measures are required to updated the map.")

        for i in range(active_idx.shape[0]):
            sensed_spots = drones_sense[active_idx[i]].copy()
            if time_point == 0:
                sensed_spots = np.delete(sensed_spots, 0, 0)
            current_measure = np.hstack((sensed_spots[-intensity_measures.shape[0]:, :], intensity_measures))
            sensed_map[active_idx[i]] = np.concatenate((sensed_map[active_idx[i]], current_measure))
            if time_point == 0:
                sensed_map[active_idx[i]] = np.delete(sensed_map[active_idx[i]], 0, 0)

        return sensed_map

    # choosing a random goal for active UAVs
    @staticmethod
    def pick_a_random_goal(previous_sensed_map=None, safe_altitudes=None, uav_class=None):
        """
        this function generates random goals for active drones to move to

        :param previous_sensed_map: most recent sensed terrain map
        :param safe_altitudes: safe altitudes associated with this all classes of UAVs
        :param uav_class: class of the current UAVs
        :return: a random goal
        """

        if previous_sensed_map is None or safe_altitudes is None or uav_class is None:
            raise ValueError(">>> Oops! Function 'pick_a_random_goal()' needs ALL its input argument to be defined (CANNOT be None).")

        # generating random points while "trying" to avoid choosing similar points
        random_goal_ind = []
        for d in range(len(previous_sensed_map)):
            rnd_ind = np.random.randint(low=0, high=len(previous_sensed_map[0]))
            if rnd_ind not in random_goal_ind:
                random_goal_ind.append(rnd_ind)
            else:
                rnd_ind = np.random.randint(low=0, high=len(previous_sensed_map[0]))
                if rnd_ind not in random_goal_ind:
                    random_goal_ind.append(rnd_ind)
                else:
                    rnd_ind = np.random.randint(low=0, high=len(previous_sensed_map[0]))
                    if rnd_ind not in random_goal_ind:
                        random_goal_ind.append(rnd_ind)
                    else:
                        rnd_ind = np.random.randint(low=0, high=len(previous_sensed_map[0]))
                        random_goal_ind.append(rnd_ind)

        counter = 0
        random_goals = []
        for current_drones_belief in previous_sensed_map:

            # evaluating the UAV half-angle
            if uav_class[counter] == 2:  # class of fixed-wings
                safe_altitude = safe_altitudes[2]
            elif uav_class[counter] == 1:  # class of multi-rotors
                safe_altitude = safe_altitudes[1]
            elif uav_class[counter] == 0:  # class of ground robots
                safe_altitude = safe_altitudes[0]
            elif uav_class[counter] == 3:  # class of hybrid UAVs
                safe_altitude = safe_altitudes[3]
            else:
                raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")

            # generating a random goal
            random_goals.append(np.hstack((current_drones_belief[random_goal_ind[counter]][0:2], safe_altitude)))

            counter += 1

        return random_goals

    # assigning specific goals to active/inactive goals
    @staticmethod
    def fighting_uav_goal_assignment(drone_stats=None, ready_idx=None, fighting_paths=None, fighting_uav_goals=None, time_step=None,
                                     safe_altitudes=None):
        """
        this function assigns goals to the fighting UAVs

        :param drone_stats: fighting UAV stats
        :param ready_idx: ready fighting UAVs indexes
        :param fighting_paths: generated fighting paths
        :param fighting_uav_goals: initialized fighting UAV goals to be stored
        :param time_step: current time point
        :param safe_altitudes: safe altitude
        :return: fighting UAV goals and index of assigned UAVs
        """

        # initial check and parsing data
        if drone_stats is None or fighting_paths is None or fighting_uav_goals is None or safe_altitudes is None:
            raise ValueError(">>> Oops! At least drone stats, fighting paths and fighting UAV goals must be specified.")

        ready_idx = np.where(drone_stats[:, 10] == 1) if ready_idx is None else ready_idx
        time_step = 0.01 if time_step is None else time_step

        uav_class = drone_stats[ready_idx, 9]
        distance_threshold = time_step * drone_stats[ready_idx, 8]

        counter = 0
        assigned_uavs_idx = []
        for d in drone_stats[ready_idx, 0:2]:

            # evaluating the UAV half-angle
            if uav_class[counter] == 2:  # class of fixed-wings
                safe_altitude = safe_altitudes[2]
            elif uav_class[counter] == 1:  # class of multi-rotors
                safe_altitude = safe_altitudes[1]
            elif uav_class[counter] == 0:  # class of ground robots
                safe_altitude = safe_altitudes[0]
            elif uav_class[counter] == 3:  # class of hybrid UAVs
                safe_altitude = safe_altitudes[3]
            else:
                raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")

            if len(fighting_paths[ready_idx[counter]]) != 0:
                this_path = fighting_paths[ready_idx[counter]][0].tolist()
                if len(this_path) != 0:
                    if np.linalg.norm(d - np.array(this_path[0])) >= distance_threshold[counter]:
                        fighting_uav_goals[ready_idx[counter], 0:3] = np.hstack((np.array(this_path[0]), safe_altitude))
                        assigned_uavs_idx.append(ready_idx[counter])
                        this_path.pop(0)
                        fighting_paths[ready_idx[counter]] = [np.array(this_path)]
                    else:
                        while np.where(np.linalg.norm(d - np.array(this_path[0])) < distance_threshold[counter]):
                            this_path.pop(0)
                            if len(this_path) == 0:
                                if len(fighting_paths[ready_idx[counter + 1]]) != 0:
                                    this_path = fighting_paths[ready_idx[counter + 1]][0].tolist()
                                else:
                                    drone_stats[ready_idx[counter], 10] = 0  # updating the fighting UAV stats to inactive NOT @ goal (retreat!)
                        if len(this_path[0]) == 0:
                            drone_stats[ready_idx[counter], 10] = 0  # updating the fighting UAV stats to inactive NOT @ goal (retreat!)
                        else:
                            fighting_uav_goals[ready_idx[counter], 0:3] = np.hstack((np.array(this_path[0]), safe_altitude))
                            this_path.pop(0)
                            fighting_paths[ready_idx[counter]] = [np.array(this_path)]

                    counter += 1
                else:
                    pass

        return fighting_uav_goals, fighting_paths, np.array(assigned_uavs_idx)

    # generate and save animation
    @staticmethod
    def generate_animation(image_folder=None, video_name=None, img_format=None):
        """
        this function generates and saves an *.avi format video animation from saved images

        :param image_folder: where images are saved? [default:: images/ in current directory]
        :param video_name: what should the video name be? [default:: animation.avi]
        :param img_format: what type of images you wanna load? [default:: *.png]
        :return:
        """

        # specifying name/format variables
        image_folder = 'images' if image_folder is None else image_folder
        video_name = 'animation.avi' if video_name is None else video_name
        img_format = '.png' if img_format is None else img_format

        # load images
        images = [img for img in os.listdir(image_folder) if img.endswith(img_format)]
        images.sort(key=lambda f: int(f.split('.')[0]))
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        # initialize video writer
        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        # create video
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # save video
        cv2.destroyAllWindows()
        video.release()
        print(">>> Done! Your animation is saved as <" + video_name + " >")

    # passing sensing info to fighting UAVs
    @staticmethod
    def pass_sensing_info(drone_loci=None, half_angles=None, uav_class=None, sensed_map=None, passed_sensing_info=None):
        """
        this function gathers sensing info and passes them to fighting UAVs

        :param drone_loci: current UAv position
        :param half_angles: camera half-angles list
        :param uav_class: current UAV class
        :param sensed_map: most recent sensed fire map
        :param passed_sensing_info: passed sensing info to fighting UAVs
        :return: passed sensing info to fighting UAVs
        """

        if drone_loci is None or half_angles is None or uav_class is None or sensed_map is None or passed_sensing_info is None:
            raise ValueError(">>> Oops! Function 'pass_sensing_info()' needs ALL of its input arguments to work!")

        current_sensing_info = []
        counter = 0
        for d in drone_loci:

            # evaluating the UAV half-angle
            if uav_class[counter] == 2:  # class of fixed-wings
                half_angle = half_angles[2]
            elif uav_class[counter] == 1:  # class of multi-rotors
                half_angle = half_angles[1]
            elif uav_class[counter] == 0:  # class of ground robots
                half_angle = half_angles[0]
            elif uav_class[counter] == 3:  # class of hybrid UAVs
                half_angle = half_angles[3]
            else:
                raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")

            # obtaining the respective FOV coordinates
            _, fov_corners_tr_bl = Utilities.fov_corners(drone_loci=d, half_angle=half_angle)
            bl_x, bl_y = fov_corners_tr_bl[1, 0], fov_corners_tr_bl[1, 1]  # bottom left coordinates
            tr_x, tr_y = fov_corners_tr_bl[0, 0], fov_corners_tr_bl[0, 1]  # top right coordinattes
            for point in sensed_map:
                x = point[0]
                y = point[1]
                if Utilities.in_fov(bl_x=bl_x, bl_y=bl_y, tr_x=tr_x, tr_y=tr_y, x=x, y=y):
                    current_sensing_info.append(point)

            if len(current_sensing_info) != 0:
                passed_sensing_info.append(np.array(current_sensing_info))
            counter += 1

        return passed_sensing_info

    # fov corners
    @staticmethod
    def fov_corners(drone_loci=None, half_angle=None):
        """
        this function returns the fov coordinates of an UAV according to its camera half-angles

        :param drone_loci: current UAV position
        :param half_angle: UAV's camera half-angles
        :return: FOV coordinates
        """

        if drone_loci is None or half_angle is None:
            raise ValueError(">>> Oops! Drone position and its camera half angles must be defined")

        half_width = drone_loci[2] * np.tan(half_angle[0])
        half_length = drone_loci[2] * np.tan(half_angle[1])

        fov_corners_1 = np.array([[drone_loci[0] + half_length, drone_loci[1] + half_width]])  # top right
        fov_corners_2 = np.array([[drone_loci[0] + half_length, drone_loci[1] - half_width]])  # bottom right
        fov_corners_3 = np.array([[drone_loci[0] - half_length, drone_loci[1] + half_width]])  # top left
        fov_corners_4 = np.array([[drone_loci[0] - half_length, drone_loci[1] - half_width]])  # bottom left

        fov_corners_all = np.concatenate((np.concatenate((fov_corners_1, fov_corners_2)), np.concatenate((fov_corners_3, fov_corners_4))))
        fov_corners_tr_bl = np.concatenate((fov_corners_1, fov_corners_4))

        return fov_corners_all, fov_corners_tr_bl

    # calculating the fov area
    @staticmethod
    def fov_area(drone_loci=None, half_angle=None, fov_corners_all=None):
        """
        this function calculates the fov area for an UAV

        :param drone_loci: current UAV position
        :param half_angle: UAV's camera half-angles
        :param fov_corners_all: FOV corners, if possible [optional]
        :return: FOV area
        """

        if fov_corners_all is None:
            if drone_loci is None or half_angle is None:
                raise ValueError(">>> Oops! Enter drone position and camera half angles as inputs.")

            fov_width = 2 * (drone_loci[2] * np.tan(half_angle[0]))
            fov_length = 2 * (drone_loci[2] * np.tan(half_angle[1]))
        else:
            fov_width = np.linalg.norm(fov_corners_all[0, :] - fov_corners_all[1, :])
            fov_length = np.linalg.norm(fov_corners_all[1, :] - fov_corners_all[3, :])

        area = fov_width * fov_length

        return area

    # check if a point is inside FOV
    @staticmethod
    def in_fov(bl_x=None, bl_y=None, tr_x=None, tr_y=None, x=None, y=None):
        """
        this function checks if a specific point is inside the FOV of an UAV. The FOV is specified by two of its coordinates

        :param bl_x: x bottom left of FOV
        :param bl_y: y bottom left of FOV
        :param tr_x: x top right of FOV
        :param tr_y: y top right of FOV
        :param x: x of the point to be checked
        :param y: y of the point to be checked
        :return: boolean flag
        """

        if bl_x is None or bl_y is None or tr_x is None or tr_y is None or x is None or y is None:
            raise ValueError(">>> Oops! Function 'in_fov()' needs ALL of its input arguments to work!")

        if x >= bl_x and x <= tr_x and y >= bl_y and y <= tr_y:
            return True
        else:
            return False

    # initialize the online visualization
    @staticmethod
    def init_visualization_2D(time_point=None, terrain_map=None, base_locations=None, base_lengths=None, base_widths=None, sensing_drone_stats=None,
                              hybrid_drone_stats=None, fighting_drone_stats=None, half_angles=None, fov_plot_flg=False, movie_generation_flg=True):
        """
        this function initializes the 2D visualization for the Heterogeneous FireFighting Robots environment

        :param time_point: current time point
        :param terrain_map: current fire map
        :param base_locations: base locations
        :param base_lengths: base lengths
        :param base_widths:base widths
        :param sensing_drone_stats: sensing UAV stats
        :param hybrid_drone_stats: hybrid UAV stats
        :param fighting_drone_stats: fighting UAV stats
        :param half_angles: all half-angles of different UAV classes
        :param fov_plot_flg: declare if you wanna see the fovs or not
        :param movie_generation_flg: movie generation flag
        :return: plot objects, also plots on current figure
        """

        if time_point is None or terrain_map is None or base_locations is None or base_lengths is None or base_widths is None \
                or sensing_drone_stats is None or hybrid_drone_stats is None or fighting_drone_stats is None:
            raise ValueError(">>> Oops! Current time point, UAV stats, fire map and base locations and sizes are required for visualization")

        if fov_plot_flg:
            if half_angles is None:
                raise ValueError(">>> Oops! If you wanna plot the FOVs, you must input the half-angles")

        # creating figure object
        fig_2d = plt.figure(1, figsize=[15, 12])  # this is where the online simulation animation will happen!
        ax_2d = plt.gca()

        # initial plots
        FireMap, = plt.plot(terrain_map[:, 0], terrain_map[:, 1], 'r*', markersize=5, label='fire spots')  # initial fire map
        PrunedFires, = [],  # initial pruned fires list
        UAVsSense, = [],  # initial UAVs sensed map
        SensingLoc, = plt.plot(sensing_drone_stats[:, 0], sensing_drone_stats[:, 1], 'bX', markersize=8, label='sensing UAVs')
        HybridLoc, = plt.plot(hybrid_drone_stats[:, 0], hybrid_drone_stats[:, 1], 'gh', markersize=10, label='hybrid UAVs')
        FightingLoc, = plt.plot(fighting_drone_stats[:, 0], fighting_drone_stats[:, 1], 'md', markersize=12, label='fighting UAVs')

        # plotting FOVs
        SensingFOVsAll, HybridFOVsAll, FightingFOVsAll = [], [], []
        if fov_plot_flg:
            for this_uav in sensing_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                SensingFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'b-')
                SensingFOVsAll.append(SensingFOV)
            for this_uav in hybrid_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                HybridFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'g-')
                HybridFOVsAll.append(HybridFOV)
            for this_uav in fighting_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                FightingFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'm-')
                FightingFOVsAll.append(FightingFOV)

        # plotting bases
        counter = 0
        for base in base_locations:
            lf_x = base[0] - base_lengths[counter] / 2
            lf_y = base[1] - base_widths[counter] / 2
            lf_xy = (lf_x, lf_y)
            string = 'base #' + str(counter + 1)
            rect = Rectangle(lf_xy, base_lengths[counter], base_widths[counter], linewidth=2, edgecolor='k', facecolor='cyan', label=string)
            ax_2d.add_patch(rect)
            counter += 1

        # figure reports and specifications
        plt.xlabel('X axis [m]')
        plt.ylabel('Y axis [m]')
        plt.title('Heterogeneous Firefighting Robots Realtime Demonstration')
        TimeCounter = plt.text(420, 10, '>> Time = ' + str(time_point) + ' [min]', fontweight='bold')
        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.legend()
        plt.show(block=False)
        plt.pause(0.000005)

        # storing the plot objects for next itteration
        plot_objects = [FireMap, PrunedFires, UAVsSense, SensingLoc, HybridLoc, FightingLoc,
                        TimeCounter, SensingFOVsAll, HybridFOVsAll, FightingFOVsAll]

        # animation movie generation
        if movie_generation_flg:
            plt.savefig('images/' + str(time_point + 1) + '.png', bbox_inches='tight')

        return plot_objects, fig_2d, ax_2d

    # online simulation visualization function
    @staticmethod
    def online_visualization_2D(plot_objects=None, fig_2d=None, ax_2d=None, time_point=None, terrain_map=None, pruned_fires=None,
                                sensing_drone_stats=None, hybrid_drone_stats=None, fighting_drone_stats=None, drones_sensed_map=None,
                                sensing_traj=None, hybrid_traj=None, fighting_traj=None, half_angles=None, traj_plot_flg=False, fov_plot_flg=False,
                                movie_generation_flg=True):
        """
        this function performs the realtime 2D visualization of the Heterogeneous FireFighting Robots environment

        :param plot_objects: all previously stored plot objects
        :param fig_2d: 2D figure object
        :param ax_2d: 2D axis object
        :param time_point: current time point
        :param terrain_map: current fire map
        :param pruned_fires: current list of pruned fires
        :param sensing_drone_stats: sensing UAV stats
        :param hybrid_drone_stats: hybrid UAV stats
        :param fighting_drone_stats: fighting UAV stats
        :param drones_sensed_map: UAVs sensed map
        :param sensing_traj: sensing UAVs trajectory
        :param hybrid_traj: hybrid UAVs trajectory
        :param fighting_traj: fighting UAVs trajectory
        :param half_angles: all half-angles of different UAV classes
        :param fov_plot_flg: declare if you wanna see the fovs or not
        :param traj_plot_flg: declare if you wanna plot the trajectories or not
        :param movie_generation_flg: movie generation flag
        :return: plot objects, also plots on current figure
        """

        if plot_objects is None or fig_2d is None or ax_2d is None:
            raise ValueError(">>> Oops! Looks like the plot objects are NOT passed. Check the initial itteration or the input arguments.")

        if time_point is None or terrain_map is None or pruned_fires is None or sensing_drone_stats is None or hybrid_drone_stats is None\
                or fighting_drone_stats is None:
            raise ValueError(">>> Oops! Current time point, UAV stats, fire map and pruned fires list are required for visualization")

        if traj_plot_flg:
            if sensing_traj is None or hybrid_traj is None or fighting_traj is None:
                raise ValueError(">>> Oops! If you wanna plot the trajectories, you must input the saved trajectories")

        if fov_plot_flg:
            if half_angles is None:
                raise ValueError(">>> Oops! If you wanna plot the FOVs, you must input the half-angles")

        # parsing the plot objects
        FireMap, = plot_objects[0],
        PrunedFires, = plot_objects[1],
        UAVsSense, = plot_objects[2],
        SensingLoc, = plot_objects[3],
        HybridLoc, = plot_objects[4],
        FightingLoc, = plot_objects[5],
        TimeCounter = plot_objects[6]
        SensingFOVsAll = plot_objects[7]
        HybridFOVsAll = plot_objects[8]
        FightingFOVsAll = plot_objects[9]

        # removing old plot objects from figure
        FireMap.remove()
        if time_point != 0:
            PrunedFires.remove()
            UAVsSense.remove()
        else:
            pass
        SensingLoc.remove()
        HybridLoc.remove()
        FightingLoc.remove()
        TimeCounter.remove()
        for i in range(len(SensingFOVsAll)):
            SensingFOV, = SensingFOVsAll[i],
            SensingFOV.remove()
        for i in range(len(HybridFOVsAll)):
            HybridFOV, = HybridFOVsAll[i],
            HybridFOV.remove()
        for i in range(len(FightingFOVsAll)):
            FightingFOV, = FightingFOVsAll[i],
            FightingFOV.remove()

        # plotting new locations
        FireMap, = plt.plot(terrain_map[:, 0], terrain_map[:, 1], 'r*', markersize=5, label='fire spots')
        PrunedFires, = plt.plot(pruned_fires[:, 0], pruned_fires[:, 1], 'ko', markersize=5, label='burnt out fires')
        UAVsSense, = plt.plot(drones_sensed_map[:, 0], drones_sensed_map[:, 1], 'b+', markersize=5, label='UAV estimations')
        SensingLoc, = plt.plot(sensing_drone_stats[:, 0], sensing_drone_stats[:, 1], 'bX', markersize=8, label='sensing UAVs')
        HybridLoc, = plt.plot(hybrid_drone_stats[:, 0], hybrid_drone_stats[:, 1], 'gh', markersize=10, label='hybrid UAVs')
        FightingLoc, = plt.plot(fighting_drone_stats[:, 0], fighting_drone_stats[:, 1], 'md', markersize=12, label='fighting UAVs')

        # plotting trajectories
        if traj_plot_flg:
            for this_traj in sensing_traj:
                temp = np.reshape(this_traj[this_traj != 0], (-1, 4))
                if len(temp) != 0:
                    plt.plot(temp[:, 0], temp[:, 1], 'b--')
            for this_traj in hybrid_traj:
                temp = np.reshape(this_traj[this_traj != 0], (-1, 4))
                if len(temp) != 0:
                    plt.plot(temp[:, 0], temp[:, 1], 'g--')
            for this_traj in fighting_traj:
                temp = np.reshape(this_traj[this_traj != 0], (-1, 4))
                if len(temp) != 0:
                    plt.plot(temp[:, 0], temp[:, 1], 'm--')

        # plotting FOVs
        SensingFOVsAll, HybridFOVsAll, FightingFOVsAll = [], [], []
        if fov_plot_flg:
            for this_uav in sensing_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                SensingFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'b-')
                SensingFOVsAll.append(SensingFOV)
            for this_uav in hybrid_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                HybridFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'g-')
                HybridFOVsAll.append(HybridFOV)
            for this_uav in fighting_drone_stats:
                # evaluating the UAV half-angle
                if this_uav[9] == 2:  # class of fixed-wings
                    half_angle = half_angles[2]
                elif this_uav[9] == 1:  # class of multi-rotors
                    half_angle = half_angles[1]
                elif this_uav[9] == 0:  # class of ground robots
                    half_angle = half_angles[0]
                elif this_uav[9] == 3:  # class of hybrid UAVs
                    half_angle = half_angles[3]
                else:
                    raise ValueError(">>> Oops! Wrong UAV class code specified. Options are integers between 0-3. Check documentation for details.")
                fov_c, _ = Utilities.fov_corners(drone_loci=this_uav, half_angle=half_angle)
                fov_edges = np.array([fov_c[0].tolist(), fov_c[1].tolist(), fov_c[3].tolist(), fov_c[2].tolist(), fov_c[0].tolist()])
                FightingFOV, = plt.plot(fov_edges[:, 0], fov_edges[:, 1], 'm-')
                FightingFOVsAll.append(FightingFOV)

        # updating figure reports and specifications
        plt.xlabel('X axis [m]')
        plt.ylabel('Y axis [m]')
        plt.title('Heterogeneous Firefighting Robots Realtime Demonstration')
        TimeCounter = plt.text(420, 10, '>> Time = ' + str(time_point + 1) + ' [min]', fontweight='bold')
        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.legend()
        plt.show(block=False)
        plt.pause(0.000005)

        # storing the plot objects for next itteration
        plot_objects = [FireMap, PrunedFires, UAVsSense, SensingLoc, HybridLoc, FightingLoc,
                        TimeCounter, SensingFOVsAll, HybridFOVsAll, FightingFOVsAll]

        # animation movie generation
        if movie_generation_flg:
            plt.savefig('images/' + str(time_point) + '.png', bbox_inches='tight')

        return plot_objects, fig_2d, ax_2d

# wildfire simulation
class HeteroFireBots_Reconn_Env_Utilities(object):
    # The function to plot the target
    # Input value: current screen, target loci infomation list, target center postion (X, Y), target size (X, Y), the flag to plot the edge or not
    # Output value: the updated list
    def target_Plot(self, screen, hospital_Font, target_Loci, current_Time):
        target_Loci_Current = target_Loci[len(target_Loci) - 1]
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

        # Append to the center location and size of target into the corresponding list
        # Content: the position of the target (X, Y), the size of the target, current time
        target_Loci.append([target_Loci_Current[0], target_Loci_Current[1], target_Loci_Current[2],
                            target_Loci_Current[3], target_Loci_Current[4], target_Loci_Current[5],
                            np.floor(current_Time / 100)])
        return target_Loci

    # The function to plot the agent base
    # Input value: current screen, agent base infomation list, the flag to plot the edge or not
    # Output value: the updated list
    def agent_Base_Plot(self, screen, agent_Base_Num, agent_Base_Loci, current_Time):
        # Search all the agent base in the list
        for i in range(agent_Base_Num):
            agent_Base_Info = agent_Base_Loci[i][len(agent_Base_Loci[i]) - 1]
            # The coordination of the upper-left corner of agent base
            agent_Base_Upper_Left_Corner = (
                agent_Base_Info[0] - agent_Base_Info[2] / 2, agent_Base_Info[1] - agent_Base_Info[3] / 2)
            agent_Base_Size = (agent_Base_Info[2], agent_Base_Info[3])
            # Plot the agent base, fill the rectangle with orange
            pygame.draw.rect(screen, (255, 225, 0), Rect(agent_Base_Upper_Left_Corner, agent_Base_Size))
            # If the edge of the agent base is enabled, plot the edge with line width 2
            if agent_Base_Info[5] == 1:
                pygame.draw.rect(screen, (0, 0, 0), Rect(agent_Base_Upper_Left_Corner, agent_Base_Size), 2)
            # Append to the center location and size of agent base into the corresponding list
            # Content: the position of the agent base (X, Y), the size of the agent base, current time
            agent_Base_Loci[i].append(
                [agent_Base_Info[0], agent_Base_Info[1], agent_Base_Info[2], agent_Base_Info[3],
                 agent_Base_Info[4], agent_Base_Info[5], np.floor(current_Time / 100)])
        return agent_Base_Loci

    # The function to mark each goal
    # The passed goal will be marked in gray, and the pending ones will be marked in red
    # Input: current screen, cureent font, user_Data_List(The whole goal list), current_Agent_State (The current goal),
    #        current patrolling goal list, move mode flag
    def goal_Marker(self, screen, font, user_Data_List, current_Agent_State, patrolling_Goal_List, move_Mode_Flag):
        # Goal display list
        text = []
        # The index of the text list
        text_Index = 0
        # Search all the goals preserved in the goal list
        for i in range(len(user_Data_List)):
            # Only plot the mouse click event (Action Type 0)
            if (user_Data_List[i][3] == 0):
                # If the given goal has been passed, mark in gary (120, 120, 120)
                if (((current_Agent_State[6] <= user_Data_List[i][4]) and current_Agent_State[14] == 0) or
                        ((user_Data_List[i] in patrolling_Goal_List) and current_Agent_State[14] == 1)):
                    # If the move_Mode_Flag is 0, mark each goal in the form like 'X_{index}'
                    if move_Mode_Flag == 0:
                        # Mark the sensor agent
                        if (current_Agent_State[8] == 0):
                            text.append(
                                font.render('P' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (0, 0, 0)))
                        # Mark the firefighter agent
                        elif (current_Agent_State[8] == 1):
                            text.append(
                                font.render('A' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (0, 0, 0)))
                        elif (current_Agent_State[8] == 2):
                            text.append(
                                font.render('H' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (0, 0, 0)))
                        # If the move_Mode_Flag is 1, only display 'X'
                    else:
                        # Mark the sensor agent
                        if (current_Agent_State[8] == 0):
                            text.append(font.render('P' + str(current_Agent_State[9]),
                                                    False, (0, 0, 0)))
                        # Mark the firefighter agent
                        elif (current_Agent_State[8] == 1):
                            text.append(font.render('A' + str(current_Agent_State[9]),
                                                    False, (0, 0, 0)))
                        elif (current_Agent_State[8] == 2):
                            text.append(font.render('H' + str(current_Agent_State[9]),
                                                    False, (0, 0, 0)))
                    # Display the goal on the screen
                    screen.blit(text[text_Index], (user_Data_List[i][0], user_Data_List[i][1]))

                # If the given goal has not been passed yet, mark in black (0, 0, 0)
                else:
                    # If the move_Mode_Flag is 0, mark each goal in the form like 'X_{index}'
                    if move_Mode_Flag == 0:
                        # Mark the sensor agent
                        if (current_Agent_State[8] == 0):
                            text.append(
                                font.render('P' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (120, 120, 120)))
                        # Mark the firefighter agent
                        elif (current_Agent_State[8] == 1):
                            text.append(
                                font.render('A' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (120, 120, 120)))
                        # Mark the hybrid agent
                        elif (current_Agent_State[8] == 2):
                            text.append(
                                font.render('H' + str(current_Agent_State[9]) + '_' + str(user_Data_List[i][4]),
                                            False, (120, 120, 120)))
                    # If the move_Mode_Flag is 1, only display 'X'
                    else:
                        # Mark the sensor agent
                        if (current_Agent_State[8] == 0):
                            text.append(font.render('P' + str(current_Agent_State[9]),
                                                    False, (120, 120, 120)))
                        # Mark the firefighter agent
                        elif (current_Agent_State[8] == 1):
                            text.append(font.render('A' + str(current_Agent_State[9]),
                                                    False, (120, 120, 120)))
                        # Mark the hybrid agent
                        elif (current_Agent_State[8] == 2):
                            text.append(font.render('H' + str(current_Agent_State[9]),
                                                    False, (120, 120, 120)))
                    # Display the goal on the screen
                    screen.blit(text[text_Index], (user_Data_List[i][0], user_Data_List[i][1]))

                # Update the index
                text_Index += 1

    # The function to control the motion of the given agent
    # Input: default agent speed (const), current agent state (Previous), move_Mode_Flag (motion mode), goal_X,
    #        goal_Y (current goal position), agent_Current_Pos_Z, agent_Init_Pos_Z (agent's flight height in current
    #        and previous moment), user_Data_List (Stored goal list), current_Goal_Index (Agent's current goal index),
    #        goal_Index (current maximum goal index), start_Flag (start moving or not), firefighter_Agent_Num,
    #        pruning_Trigger, battary_para, original_Agent_State, patrolling_Goal_List, current time
    # Output: current agent state (Updated), agent_Current_Pos_Z, agent_Init_Pos_Z (agent's flight height in current
    #         and previous moment), Updated goal index
    def agent_Motion_Controller(self, agent_Speed, current_Agent_State, move_Mode_Flag, goal_X, goal_Y,
                                agent_Current_Pos_Z,
                                agent_Init_Pos_Z, user_Data_List, current_Goal_Index, goal_Index, start_Flag,
                                firefighter_Agent_Num, pruning_Trigger, battery_para, original_Agent_State,
                                patrolling_Goal_List, waiting_Time_List, current_Time):
        # If its value of the move mode flag is 1, the agent will fly by one step when clicking
        if move_Mode_Flag == 1:
            # Update the current time
            current_Agent_State[7] = current_Time
            # Current goal will be left as 0
            current_Agent_State[6] = 0

            # Acquire the postion of the current goal
            current_Goal_X = goal_X
            current_Goal_Y = goal_Y

            # Calculate the planar velocity direction vector
            vector_X = current_Goal_X - current_Agent_State[0]
            vector_Y = current_Goal_Y - current_Agent_State[1]
            vector_Amp = np.sqrt(vector_X ** 2 + vector_Y ** 2)

            # Calculate the agent speed
            current_Agent_State[3] = agent_Speed * vector_X / vector_Amp
            current_Agent_State[4] = agent_Speed * vector_Y / vector_Amp
            current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z
            # Update the agent postion
            current_Agent_State[0] = np.floor(current_Agent_State[0] + current_Agent_State[3])
            current_Agent_State[1] = np.floor(current_Agent_State[1] + current_Agent_State[4])
            current_Agent_State[2] = agent_Current_Pos_Z

            # Update the whole distance
            current_Agent_State[10] += agent_Speed

            # Update the current flight height
            agent_Init_Pos_Z = agent_Current_Pos_Z

        # If its value of the move mode flag is 0, the agent will automatically fly towards the goal
        elif move_Mode_Flag == 0:
            # Update the current state of the agent
            if current_Agent_State[13] == 1:
                # Case 1: if the goal index is 0, the agent will be silent, the current time in the array will be updated
                if (current_Agent_State[6] == 0):
                    # Update the current time
                    current_Agent_State[7] = current_Time
                    # If the start flag is 1 and the current goal is 0, set the target as 1
                    if start_Flag == 1:
                        current_Agent_State[6] = 1
                # Case 2: if the agent is moving and still far away from the goal, fly towards the goal
                elif start_Flag == 1:
                    # Acquire the postion of the current goal
                    # Normal Mode: Extract the info from the user_Data_List
                    if current_Agent_State[14] == 0:
                        current_Goal_X = user_Data_List[current_Goal_Index][0]
                        current_Goal_Y = user_Data_List[current_Goal_Index][1]
                    # Patrolling Mode: Extract the info from the patrolling_Goal_List
                    else:
                        current_Goal_X = patrolling_Goal_List[current_Agent_State[15]][0]
                        current_Goal_Y = patrolling_Goal_List[current_Agent_State[15]][1]

                    # Calculate the planar velocity direction vector
                    vector_X = current_Goal_X - current_Agent_State[0]
                    vector_Y = current_Goal_Y - current_Agent_State[1]
                    vector_Amp = np.sqrt(vector_X ** 2 + vector_Y ** 2)

                    # If the distance between the agent and the goal is larger than 10, fly towards the goal
                    if (vector_Amp >= agent_Speed):
                        # Calculate the agent speed
                        current_Agent_State[3] = agent_Speed * vector_X / vector_Amp
                        current_Agent_State[4] = agent_Speed * vector_Y / vector_Amp
                        current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z
                        # Update the agent postion
                        current_Agent_State[0] = np.floor(current_Agent_State[0] + current_Agent_State[3])
                        current_Agent_State[1] = np.floor(current_Agent_State[1] + current_Agent_State[4])
                        current_Agent_State[2] = agent_Current_Pos_Z

                        # Update the current time
                        current_Agent_State[7] = current_Time

                        # Update the whole distance
                        current_Agent_State[10] += agent_Speed

                        # Update the waiting time
                        current_Agent_State[11] += 0

                        trigger_Index = current_Agent_State[9] - 1 + firefighter_Agent_Num * (
                                current_Agent_State[8] - 1)
                        if pruning_Trigger[trigger_Index] == 2:
                            pruning_Trigger[trigger_Index] = 0

                    # If not, the agent reaches the goal, go to the next target or silent
                    else:
                        # Switch the status of the firefighter agent when it arrives the goal
                        if (current_Agent_State[8] == 1) or (
                                (current_Agent_State[8] == 2) and (current_Agent_State[12] > 0)):
                            trigger_Index = current_Agent_State[9] - 1 + firefighter_Agent_Num * (
                                    current_Agent_State[8] - 1)
                            if pruning_Trigger[trigger_Index] == 0:
                                pruning_Trigger[trigger_Index] = 1

                        # Determine the next goal
                        # Normal Mode: Extract the info from the user_Data_List
                        if current_Agent_State[14] == 0:
                            # Determine whether the current goal is the last one
                            if (current_Agent_State[6] < goal_Index):
                                current_Agent_State[6] += 1
                                # Search for the corresponding index of the new goal in the user data list
                                for i in range(len(user_Data_List)):
                                    if user_Data_List[i][4] == current_Agent_State[6]:
                                        # Update the current goal index
                                        current_Goal_Index = i
                                        break

                                # Switch the status of the firefighter agent when it arrives the goal
                                if (current_Agent_State[8] == 1) or (current_Agent_State[8] == 2):
                                    trigger_Index = current_Agent_State[9] - 1 + firefighter_Agent_Num * (
                                            current_Agent_State[8] - 1)
                                    if pruning_Trigger[trigger_Index] == 2:
                                        pruning_Trigger[trigger_Index] = 0

                                if current_Agent_State[8] == 1:
                                    waiting_Time_List[current_Agent_State[9] - 1] = 0

                            else:
                                if current_Agent_State[8] == 1:
                                    if current_Agent_State[13] == 1 and waiting_Time_List[current_Agent_State[9] - 1] == 0:
                                        waiting_Time_List[current_Agent_State[9] - 1] = current_Time
                                        # Clear the speed
                                        current_Agent_State[3] = 0
                                        current_Agent_State[4] = 0
                                        current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z

                                        # Update the whole distance
                                        current_Agent_State[10] += np.sqrt(
                                            (current_Agent_State[0] - current_Goal_X) ** 2
                                            + (current_Agent_State[1] - current_Goal_Y) ** 2)

                                        # Set the agent position overlapping the last goal
                                        current_Agent_State[0] = current_Goal_X
                                        current_Agent_State[1] = current_Goal_Y
                                        current_Agent_State[2] = agent_Current_Pos_Z
                                        current_Agent_State[6] = current_Agent_State[6]

                                        # Update the current time
                                        current_Agent_State[7] = current_Time

                                        # Update the waiting time
                                        current_Agent_State[11] += 1

                                    elif current_Agent_State[13] == 1 and (current_Time - waiting_Time_List[current_Agent_State[9] - 1]) >= 3000:
                                        waiting_Time_List[current_Agent_State[9] - 1] = 0
                                        current_Agent_State[13] = 2
                                    else:
                                        # Clear the speed
                                        current_Agent_State[3] = 0
                                        current_Agent_State[4] = 0
                                        current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z

                                        # Update the whole distance
                                        current_Agent_State[10] += np.sqrt(
                                            (current_Agent_State[0] - current_Goal_X) ** 2
                                            + (current_Agent_State[1] - current_Goal_Y) ** 2)

                                        # Set the agent position overlapping the last goal
                                        current_Agent_State[0] = current_Goal_X
                                        current_Agent_State[1] = current_Goal_Y
                                        current_Agent_State[2] = agent_Current_Pos_Z
                                        current_Agent_State[6] = current_Agent_State[6]

                                        # Update the current time
                                        current_Agent_State[7] = current_Time

                                        # Update the waiting time
                                        current_Agent_State[11] += 1
                                else:
                                    # Clear the speed
                                    current_Agent_State[3] = 0
                                    current_Agent_State[4] = 0
                                    current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z

                                    # Update the whole distance
                                    current_Agent_State[10] += np.sqrt(
                                        (current_Agent_State[0] - current_Goal_X) ** 2
                                        + (current_Agent_State[1] - current_Goal_Y) ** 2)

                                    # Set the agent position overlapping the last goal
                                    current_Agent_State[0] = current_Goal_X
                                    current_Agent_State[1] = current_Goal_Y
                                    current_Agent_State[2] = agent_Current_Pos_Z
                                    current_Agent_State[6] = current_Agent_State[6]

                                    # Update the current time
                                    current_Agent_State[7] = current_Time

                                    # Update the waiting time
                                    current_Agent_State[11] += 1
                        # Patrolling Mode: Extract the info from the patrolling_Goal_List
                        else:
                            current_Agent_State[15] = int((current_Agent_State[15] + 1) % len(patrolling_Goal_List))

                    # Update the current flight height
                    agent_Init_Pos_Z = agent_Current_Pos_Z

                current_Distance = np.sqrt((current_Agent_State[0] - original_Agent_State[0]) ** 2 +
                                           (current_Agent_State[1] - original_Agent_State[1]) ** 2)
                if battery_para[0] <= (battery_para[1] * (current_Agent_State[10] + current_Distance)
                                       + battery_para[2] * current_Agent_State[11]):
                    current_Agent_State[13] = 0

            else:
                if (current_Agent_State[13] == 0) or (current_Agent_State[13] == 2):
                    # Acquire the postion of the current goal
                    current_Goal_X = original_Agent_State[0]
                    current_Goal_Y = original_Agent_State[1]

                    # Calculate the planar velocity direction vector
                    vector_X = current_Goal_X - current_Agent_State[0]
                    vector_Y = current_Goal_Y - current_Agent_State[1]
                    vector_Amp = np.sqrt(vector_X ** 2 + vector_Y ** 2)

                    # If the distance between the agent and the goal is larger than 10, fly towards the goal
                    if (vector_Amp >= agent_Speed):
                        # Calculate the agent speed
                        current_Agent_State[3] = agent_Speed * vector_X / vector_Amp
                        current_Agent_State[4] = agent_Speed * vector_Y / vector_Amp
                        current_Agent_State[5] = agent_Current_Pos_Z - agent_Init_Pos_Z
                        # Update the agent postion
                        current_Agent_State[0] = np.floor(current_Agent_State[0] + current_Agent_State[3])
                        current_Agent_State[1] = np.floor(current_Agent_State[1] + current_Agent_State[4])
                        current_Agent_State[2] = agent_Current_Pos_Z

                        # Update the current time
                        current_Agent_State[7] = current_Time

                        # Update the whole distance
                        current_Agent_State[10] += agent_Speed

                        # Update the waiting time
                        current_Agent_State[11] += 0

                    # If not, the agent reaches the goal, go to the next target or silent
                    else:
                        # Set the agent position overlapping the last goal
                        original_Agent_State[2] = current_Agent_State[2]

                        # Set the agent position overlapping the last goal
                        original_Agent_State[6] = current_Agent_State[6]

                        # Update the current time
                        original_Agent_State[7] = current_Time

                        if current_Agent_State[13] == 0:
                            # Preserve the patrolling loop info
                            original_Agent_State[14] = current_Agent_State[14]
                            original_Agent_State[15] = current_Agent_State[15]
                        else:
                            start_Flag = 0
                            original_Agent_State[14] = 0
                            original_Agent_State[15] = 0

                        # Update the waiting time
                        for i in range(len(current_Agent_State)):
                            current_Agent_State[i] = original_Agent_State[i]

        return current_Agent_State, agent_Current_Pos_Z, agent_Init_Pos_Z, current_Goal_Index, pruning_Trigger, start_Flag, waiting_Time_List

    # The function fire_Data_Storage is used to separate the fire spots in different regions for storage
    # Input: The number of the ignited fire spots, the whole fire map, the generated fire spot, the size of the
    #         simulation environment, the number of the fire regions and the previous fire state list and fire map,
    #         the previous onFire_List
    # Output: The updated fire map, the fire state list for storage, the updated onFire_List
    def fire_Data_Storage(self, num_ign_points, fire_States_List, new_fire_front, world_Size,
                          fireSpots_Num, fire_Current_Map, current_Time, onFire_List, target_onFire_list, target_onFire_Flag, target_info, spec_flag, fire_turnon_flag):
        # Initialize the list to store the fire front that locates inside the target region
        target_new_firefront = []
        for i in range(len(target_onFire_list)):
            target_new_firefront.append([])
            for j in range(len(target_onFire_list[i])):
                target_new_firefront[i].append(target_onFire_list[i][j][len(target_onFire_list[i][j]) - 1])

        if new_fire_front.shape[0] > 0:
            # Write the fire spot into the current world map list
            for i in range(new_fire_front.shape[0]):
                # # Ensure that all the fire spots to be displayed must be within the window scope
                if ((new_fire_front[i][0] <= (world_Size - 1)) and (new_fire_front[i][1] <= (world_Size - 1))
                        and (new_fire_front[i][0] >= 0) and (new_fire_front[i][1] >= 0)):
                    fire_Current_Map[int(new_fire_front[i][0])][int(new_fire_front[i][1])] += new_fire_front[i][2]

                    # If the new fire front points is not included in the current onFire_List, add it into the list
                    if ([int(new_fire_front[i][0]), int(new_fire_front[i][1])] not in onFire_List):
                        onFire_List.append([int(new_fire_front[i][0]), int(new_fire_front[i][1])])

                        # Determine whether the new fire fronts locate inside the target region
                        for i1 in range(len(target_onFire_list)):
                            for j1 in range(len(target_onFire_list[i1])):
                                if (int(new_fire_front[i][0]) > (target_info[i1][j1][0] - target_info[i1][j1][2] / 2)) and (int(new_fire_front[i][0]) < (target_info[i1][j1][0] + target_info[i1][j1][2] / 2)) and \
                                (int(new_fire_front[i][1]) > (target_info[i1][j1][1] - target_info[i1][j1][3] / 2)) and (int(new_fire_front[i][1]) < (target_info[i1][j1][1] + target_info[i1][j1][3] / 2)):
                                    target_new_firefront[i1][j1] += 1
                                    target_onFire_Flag[i1][j1] = 1

            if spec_flag == 0:
                # Write the fire spot into the current world map list
                for i in range(fireSpots_Num):
                    for j in range(num_ign_points):
                        fire_States_List[i].append([new_fire_front[i * num_ign_points + j][0],
                                                    new_fire_front[i * num_ign_points + j][1],
                                                    new_fire_front[i * num_ign_points + j][2], current_Time])
            else:
                # Write the fire spot into the current world map list
                count = 0
                for i in range(fireSpots_Num):
                    if fire_turnon_flag[i] == 1:
                        for j in range(num_ign_points[i]):
                            fire_States_List[i].append([new_fire_front[count + j][0],
                                                        new_fire_front[count + j][1],
                                                        new_fire_front[count + j][2], current_Time])
                        count += num_ign_points[i]

        # Write the number of fire fronts inside the target into the storage list
        for i1 in range(len(target_onFire_list)):
            for j1 in range(len(target_onFire_list[i1])):
                target_onFire_list[i1][j1].append(target_new_firefront[i1][j1])

        return fire_Current_Map, fire_States_List, onFire_List, target_onFire_list, target_onFire_Flag

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

    # Calculate the center of mass for the given sensed fire spot
    # Input: The sensed fire spot data
    @staticmethod
    def center_Of_Mass_Calculate(sensed_Data):
        # Convert into the array
        sensed_Data = np.array(sensed_Data)
        # Calculate the coordinates of the CoM
        com_X = sum(sensed_Data[:, 0]) / sensed_Data.shape[0]
        com_Y = sum(sensed_Data[:, 1]) / sensed_Data.shape[0]
        # Calculate the maximum and average fire intensity
        max_Intensity = max(sensed_Data[:, 2])
        avg_Intensity = sum(sensed_Data[:, 2]) / sensed_Data.shape[0]
        # Calculate the average velocity
        avg_Velocity = sum(sensed_Data[:, 3]) / sensed_Data.shape[0]
        return [com_X, com_Y, max_Intensity, avg_Velocity * max_Intensity / avg_Intensity]

    # Fire transfer speed calculation with the simplified FARSITE
    @staticmethod
    def fire_Propagation_Velocity(fire_Loci, geo_phys_info, world_Size):
        # Extracting the data
        [x, y] = fire_Loci
        spread_rate = geo_phys_info['spread_rate']

        wind_speed = geo_phys_info['wind_speed']
        wind_direction = geo_phys_info['wind_direction']

        # Extracting the required information
        R = spread_rate[np.array(x).astype(int), np.array(y).astype(int)]

        U = wind_speed[np.random.randint(low=0, high=world_Size)][0]
        Theta = wind_direction[np.random.randint(low=0, high=world_Size)][0]
        # current_geo_phys_info = np.array([R, U, Theta])  # storing GP information

        # Calculate the necessary parameters: LB, HB, C
        LB = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397
        HB = (LB + np.sqrt(np.absolute(np.power(LB, 2) - 1))) / (LB - np.sqrt(np.absolute(np.power(LB, 2) - 1)))
        C = 0.5 * (R - (R / HB))

        # Calculate the velocity
        x_diff = C * np.sin(Theta)
        y_diff = C * np.cos(Theta)
        velocity = np.sqrt(x_diff ** 2 + y_diff ** 2)

        return velocity

    # Acquire the fire spot information in the given region
    # Input: fire_Map, current agent state, the agent's FOV, geometric_physics info, sensed_List, window size
    # Output: fire sensed map (for agent status recording), CoM info, the list of the coordinates of the sensed points
    def fire_Sensing(self, fire_map, current_Agent_State, agent_FOV, geo_phys_info, onFire_List, sensed_List, world_Size, num_ign_points, spec_flag, fire_turnon_flag, height_info):
        # Initialize the list to store the sensed fire state
        fire_Sensed_Map = []

        # Initialize the height info (Upper and lower bound, current height) from the external height info list
        [lower_bound, upper_bound, current_height] = height_info

        # If the lower_bound equals to the upper bound (Fixed perception height), set the confidence level as 1
        if lower_bound == upper_bound:
            confidence_level = 1
        # If not, compute the confidence level (0.4 - 1.0) in proportion to the height range (Lower - Upper Bound)
        elif lower_bound < upper_bound:
            confidence_level = 1 - (current_height - lower_bound) / (upper_bound - lower_bound) * 0.6

        # Calculate the size of the searching scope
        searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State[2]
        searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State[2]
        # The coordination of the upper-left corner of the agent searching scope
        (tl_x, tl_y) = (current_Agent_State[0] - searching_Scope_X / 2,
                        current_Agent_State[1] - searching_Scope_Y / 2)

        # The coordination of the lower-right corner of the agent searching scope
        (br_x, br_y) = (current_Agent_State[0] + searching_Scope_X / 2,
                        current_Agent_State[1] + searching_Scope_Y / 2)

        if spec_flag == 0:
            # Search for the current fire map, determine whether the given fire spot locates within the searching scope
            raw_sensed_idx = np.intersect1d(np.argwhere(fire_map[:, 0] <= br_x), np.argwhere(fire_map[:, 0] >= tl_x))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[:, 1] <= br_y))
            raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[:, 1] >= tl_y))
            # Apply the stochastic perception
            raw_sensed_idx = np.random.choice(raw_sensed_idx, int(round(confidence_level * len(raw_sensed_idx))), replace=False)
            raw_sensed_list = fire_map[raw_sensed_idx, :]
            if fire_turnon_flag == 1 and len(raw_sensed_list[:, 0]) > 0:
                geo_phys_info_temp = geo_phys_info
                fire_Velocity = self.fire_Propagation_Velocity([raw_sensed_list[:, 0], raw_sensed_list[:, 1]],
                                                               geo_phys_info_temp,
                                                               world_Size)

                fire_Sensed_Map = np.zeros((len(raw_sensed_list[:, 0]), 4), dtype=float)
                fire_Sensed_Map[:, 0] = raw_sensed_list[:, 0]
                fire_Sensed_Map[:, 1] = raw_sensed_list[:, 1]
                fire_Sensed_Map[:, 2] = raw_sensed_list[:, 2]
                fire_Sensed_Map[:, 3] = fire_Velocity

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

        else:
            # Search for the current fire map, determine whether the given fire spot locates within the searching scope
            for i in range(len(fire_map)):
                # Search for the current fire map, determine whether the given fire spot locates within the searching scope
                raw_sensed_idx = np.intersect1d(np.argwhere(fire_map[i][:, 0] <= br_x),
                                                np.argwhere(fire_map[i][:, 0] >= tl_x))
                raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[i][:, 1] <= br_y))
                raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[i][:, 1] >= tl_y))
                # Apply the stochastic perception
                raw_sensed_idx = np.random.choice(raw_sensed_idx, int(round(confidence_level * len(raw_sensed_idx))), replace=False)
                raw_sensed_list = fire_map[i][raw_sensed_idx, :]
                if fire_turnon_flag[i] == 1 and len(raw_sensed_list[:, 0]) > 0:
                    geo_phys_info_temp = geo_phys_info[i]
                    fire_Velocity = self.fire_Propagation_Velocity([raw_sensed_list[:, 0], raw_sensed_list[:, 1]],
                                                                   geo_phys_info_temp,
                                                                   world_Size)

                    fire_Sensed_Map = np.zeros((len(raw_sensed_list[:, 0]), 4), dtype=float)
                    fire_Sensed_Map[:, 0] = raw_sensed_list[:, 0]
                    fire_Sensed_Map[:, 1] = raw_sensed_list[:, 1]
                    fire_Sensed_Map[:, 2] = raw_sensed_list[:, 2]
                    fire_Sensed_Map[:, 3] = fire_Velocity

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

        # If the sensed fire spot list is not null, calculate its center of mass
        CoM_Info = []
        if len(fire_Sensed_Map) > 0:
            CoM_Info = self.center_Of_Mass_Calculate(fire_Sensed_Map)

        return fire_Sensed_Map, CoM_Info, sensed_List

    # Pruning the fire with the firefighter agents, if given fire spots locate within the firefighter agents' scope,
    # delete them from the onFire and sensed list, add them into the pruned list, create the pruned list with time stamp
    # for the current agent (For data storage)
    # Input: fire_Map, current agent state, the agent's FOV, onFire_List, sensed_List, pruned_List, new fire front list
    # Output: fire_Pruned_Map (for agent status recording), the updated fire_map, onFire_List, sensed_List, pruned_List
    def fire_Pruning(self, fire_map, current_Agent_State, agent_FOV, onFire_List, sensed_List, pruned_List,
                     new_fire_front, target_onFire_list, target_info, confidence_level):
        # Initialize the list to store the sensed fire state
        fire_Pruned_Map = []

        # Calculate the size of the searching scope
        searching_Scope_X = 2 * np.tan(agent_FOV[0]) * current_Agent_State[2]
        searching_Scope_Y = 2 * np.tan(agent_FOV[1]) * current_Agent_State[2]

        # The coordination of the upper-left corner of the agent searching scope
        (tl_x, tl_y) = (current_Agent_State[0] - searching_Scope_X / 2,
                        current_Agent_State[1] - searching_Scope_Y / 2)

        # The coordination of the lower-right corner of the agent searching scope
        (br_x, br_y) = (current_Agent_State[0] + searching_Scope_X / 2,
                        current_Agent_State[1] + searching_Scope_Y / 2)
        # Temporary list to store the fire front that may be pruned
        temp_list = []
        # sensed list flag, if there is any points that is included in the sensed list, this flag will become 1
        sensed_flag = 0

        # Search for the current fire map, determine whether the given fire spot locates within the searching scope
        raw_sensed_idx = np.intersect1d(np.argwhere(fire_map[:, 0] <= br_x), np.argwhere(fire_map[:, 0] >= tl_x))
        raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[:, 1] <= br_y))
        raw_sensed_idx = np.intersect1d(raw_sensed_idx, np.argwhere(fire_map[:, 1] >= tl_y))
        # Apply the stochastic pruning
        raw_sensed_idx = np.random.choice(raw_sensed_idx, int(round(confidence_level * len(raw_sensed_idx))),
                                          replace=False)
        temp_list = fire_map[raw_sensed_idx, :]
        if len(temp_list) > 0:
            int_pruned_list = np.zeros((len(temp_list[:, 0]), 2), dtype=int)
            int_pruned_list[:, 0] = temp_list[:, 0].astype(int)
            int_pruned_list[:, 1] = temp_list[:, 1].astype(int)
            uni_int_pruned_list = np.unique(int_pruned_list, axis=0)

            sensed_List_copy = sensed_List.copy()

            if len(sensed_List_copy) > 0:
                sensed_List_copy[0:0] = list(uni_int_pruned_list).copy()
                if len(np.unique(np.array(sensed_List_copy), axis=0).tolist()) < (len(sensed_List) + len(uni_int_pruned_list)):
                    sensed_flag = 1
            else:
                sensed_flag = 1

            # The pruning agent could only put out fire region that contains the sensed fire fronts
            if sensed_flag == 1:
                for i in range(len(temp_list)):
                    # If the new fire front points is not included in the current onFire_List, add it into the list
                    if (([int(temp_list[i][0]), int(temp_list[i][1])] not in pruned_List) and
                            ([int(temp_list[i][0]), int(temp_list[i][1])] in onFire_List)):
                        onFire_List.remove([int(temp_list[i][0]), int(temp_list[i][1])])
                        if ([int(temp_list[i][0]), int(temp_list[i][1])] in sensed_List):
                            sensed_List.remove([int(temp_list[i][0]), int(temp_list[i][1])])

                        # Determine whether the new fire fronts locate inside the target region
                        for i1 in range(len(target_onFire_list)):
                            for j1 in range(len(target_onFire_list[i1])):
                                if (int(temp_list[i][0]) > (target_info[i1][j1][0] - target_info[i1][j1][2] / 2)) and (int(temp_list[i][0]) < (target_info[i1][j1][0] + target_info[i1][j1][2] / 2)) and \
                                (int(temp_list[i][1]) > (target_info[i1][j1][1] - target_info[i1][j1][3] / 2)) and (int(temp_list[i][1]) < (target_info[i1][j1][1] + target_info[i1][j1][3] / 2)):
                                    target_onFire_list[i1][j1][len(target_onFire_list[i1][j1]) - 1] -= 1

                        pruned_List.append([int(temp_list[i][0]), int(temp_list[i][1])])
                        fire_Pruned_Map.append([temp_list[i][0], temp_list[i][1], temp_list[i][2]])

        return fire_Pruned_Map, fire_map, onFire_List, sensed_List, pruned_List, new_fire_front, target_onFire_list, sensed_flag

    # Display the battery capacity and water tank info
    # Input: current screen, current states of each agent, display font, battery parameter, user_Data_List,
    #        index_1st, goal_Index_List, the world size
    def side_Bar_Display(self, screen, current_Agent_State_List, font, font_Bold, font_Title, battery_para, user_Data_List, index_Next,
                         goal_Index_List, world_Size):
        column_Size = 30

        # Table 1: Battery Capacity Info
        screen.blit(font_Title.render('Battery Capacity Info: ', True, (0, 0, 0)), (world_Size + 20, 20))
        # Title for each column
        # Column 1: Agent
        screen.blit(font_Bold.render('Agent', True, (0, 0, 0)), (world_Size + 20, 60))
        # Column 2: Remaining Energy
        screen.blit(font_Bold.render('Remaining', True, (0, 0, 0)), (world_Size + 150, 60 - 10))
        screen.blit(font_Bold.render('Energy', True, (0, 0, 0)), (world_Size + 150, 60 + 10))
        # Column 3: Estimated Energy Till Next Goal
        screen.blit(font_Bold.render('Estimated Energy', True, (0, 0, 0)), (world_Size + 280, 60 - 10))
        screen.blit(font_Bold.render('Till Next Goal', True, (0, 0, 0)), (world_Size + 280, 60 + 10))
        # Column 4: Current agent flight height
        screen.blit(font_Bold.render('Current', True, (0, 0, 0)), (world_Size + 470, 60 - 10))
        screen.blit(font_Bold.render('Height', True, (0, 0, 0)), (world_Size + 470, 60 + 10))

        # The index of the Battery Capacity Info list
        text_Index = 1

        # Display the battery capacity info for each agent
        for i in range(len(current_Agent_State_List)):
            remaining_Battery = int(battery_para[i][0] - battery_para[i][1] * current_Agent_State_List[i][10] \
                                    - battery_para[i][2] * current_Agent_State_List[i][11])
            if current_Agent_State_List[i][8] == 0:
                screen.blit(font_Bold.render('Perception ' + str(current_Agent_State_List[i][9]) + ': ',
                                        True, (0, 0, 0)), (world_Size + 20, 70 + text_Index * column_Size))
            elif current_Agent_State_List[i][8] == 1:
                screen.blit(font_Bold.render('Action ' + str(current_Agent_State_List[i][9]) + ': ',
                                        True, (0, 0, 0)), (world_Size + 20, 70 + text_Index * column_Size))
            elif current_Agent_State_List[i][8] == 2:
                screen.blit(font_Bold.render('Hybrid ' + str(current_Agent_State_List[i][9]) + ': ',
                                        True, (0, 0, 0)), (world_Size + 20, 70 + text_Index * column_Size))
            screen.blit(font.render(str(remaining_Battery), True, (0, 0, 0)),
                        (world_Size + 150, 70 + text_Index * column_Size))

            if int(index_Next[i]) >= 0:
                sum_Distance = 0
                for j in range(int(index_Next[i]), len(goal_Index_List[i]) - 1):
                    sum_Distance += np.sqrt((user_Data_List[i][j + 1][0] - user_Data_List[i][j][0]) ** 2 +
                                            (user_Data_List[i][j + 1][1] - user_Data_List[i][j][1]) ** 2)

                remaining_Dis = np.sqrt(
                    (current_Agent_State_List[i][0] - user_Data_List[i][int(index_Next[i])][0]) ** 2 +
                    (current_Agent_State_List[i][1] - user_Data_List[i][int(index_Next[i])][1]) ** 2)
                screen.blit(
                    font.render(str(remaining_Battery - int(battery_para[i][1] * (remaining_Dis + sum_Distance))),
                                True, (0, 0, 0)),
                    (world_Size + 280, 70 + text_Index * column_Size))
            else:
                screen.blit(font.render('N/A', True, (0, 0, 0)),
                            (world_Size + 280, 70 + text_Index * column_Size))

            # Display the current height for sensing and hybrid agents. For the pruning agents, the height display is disabled
            if current_Agent_State_List[i][8] == 0 or current_Agent_State_List[i][8] == 2:
                screen.blit(font.render(str(current_Agent_State_List[i][2]), True, (0, 0, 0)),
                            (world_Size + 470, 70 + text_Index * column_Size))
            else:
                screen.blit(font.render('N/A', True, (0, 0, 0)), (world_Size + 470, 70 + text_Index * column_Size))

            text_Index += 1

        # Table 2: Water Tank Info
        screen.blit(font_Title.render('Water Tank Info: ', True, (0, 0, 0)), (world_Size + 20, 100 + text_Index * column_Size))
        # Title for each column
        # Column 1: Agent
        screen.blit(font_Bold.render('Agent', True, (0, 0, 0)), (world_Size + 20, 140 + text_Index * column_Size))
        # Column 2: Remaining Pruning Times
        screen.blit(font_Bold.render('Remaining Pruning Times', True, (0, 0, 0)),
                    (world_Size + 150, 140 + text_Index * column_Size))

        # Display the water tank info for each agent
        for i in range(len(current_Agent_State_List)):
            if current_Agent_State_List[i][8] == 1:
                screen.blit(font_Bold.render('Action ' + str(current_Agent_State_List[i][9]) + ': ',
                                        True, (0, 0, 0)), (world_Size + 20, 180 + text_Index * column_Size))
                screen.blit(font.render(str(int(current_Agent_State_List[i][12])), False, (0, 0, 0)),
                            (world_Size + 250, 180 + text_Index * column_Size))
                text_Index += 1

            elif current_Agent_State_List[i][8] == 2:
                screen.blit(font_Bold.render('Hybrid ' + str(current_Agent_State_List[i][9]) + ': ',
                                        True, (0, 0, 0)), (world_Size + 20, 180 + text_Index * column_Size))
                screen.blit(font.render(str(int(current_Agent_State_List[i][12])), False, (0, 0, 0)),
                            (world_Size + 250, 180 + text_Index * column_Size))
                text_Index += 1
        pos = (world_Size + 20, 220 + text_Index * column_Size)
        return pos

    # External function: plot the dashline between two given points
    def draw_dashed_line(serf, screen, color, start_pos, end_pos, width=1, dash_length=10):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = dash_length

        if (x1 == x2):
            ycoords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
            xcoords = [x1] * len(ycoords)
        elif (y1 == y2):
            xcoords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
            ycoords = [y1] * len(xcoords)
        else:
            a = abs(x2 - x1)
            b = abs(y2 - y1)
            c = round(np.sqrt(a ** 2 + b ** 2))
            dx = dl * a / c
            dy = dl * b / c

            xcoords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
            ycoords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]

        next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
        last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
        for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
            start = (round(x1), round(y1))
            end = (round(x2), round(y2))
            pygame.draw.line(screen, color, start, end, width)

    # Trajectory plot
    # Input: current screen, current state for all agents, the info of the goal for each agent, the given agent's
    #        patrolling_Goal_List, the agent's color
    # Output: index_1st
    def traj_Plot(self, screen, user_Data_List, current_Agent_State, patrolling_Goal_List, color):
        # The index of the agent's first goal
        index_1st = -1
        # The previous index of the agent goal
        index_Start = -1

        # Normal mode: plot the rest part of the trajectory
        if current_Agent_State[14] == 0:
            # Search all the goals preserved in the goal list
            for i in range(len(user_Data_List)):
                # Only plot the mouse click event (Action Type 0)
                if (user_Data_List[i][3] == 0):
                    # If the given goal has not been passed, plot the dashline
                    if current_Agent_State[6] <= user_Data_List[i][4]:
                        # Plot the dashline trajectory between the current agent position and the 1st goal
                        if index_Start == -1:
                            index_1st = i
                            start_pos = (int(current_Agent_State[0]), int(current_Agent_State[1]))
                            end_pos = (user_Data_List[index_1st][0], user_Data_List[index_1st][1])
                            self.draw_dashed_line(screen, color, start_pos, end_pos, width=2, dash_length=5)
                            index_Start = i
                        # Plot the dashline trajectory between the remaining goals
                        else:
                            start_pos = (int(user_Data_List[index_Start][0]), int(user_Data_List[index_Start][1]))
                            end_pos = (int(user_Data_List[i][0]), int(user_Data_List[i][1]))
                            self.draw_dashed_line(screen, color, start_pos, end_pos, width=2, dash_length=5)
                            index_Start = i

        # Patrolling mode: go over all the elements in the patrolling goal list
        else:
            for i in range(len(patrolling_Goal_List) - 1):
                start_pos = (int(patrolling_Goal_List[i][0]), int(patrolling_Goal_List[i][1]))
                end_pos = (int(patrolling_Goal_List[i + 1][0]), int(patrolling_Goal_List[i + 1][1]))
                self.draw_dashed_line(screen, color, start_pos, end_pos, width=2, dash_length=5)

            start_pos = (int(patrolling_Goal_List[len(patrolling_Goal_List) - 1][0]),
                         int(patrolling_Goal_List[len(patrolling_Goal_List) - 1][1]))
            end_pos = (int(patrolling_Goal_List[0][0]), int(patrolling_Goal_List[0][1]))
            self.draw_dashed_line(screen, color, start_pos, end_pos, width=2, dash_length=5)
        return index_1st

    # Determine whether the given point inside the base
    def in_Agent_Base_Region(self, goal_X, goal_Y, agent_Base_Num, agent_Base_Loci_Full):
        in_Base_Flag = False
        base_Index = -1
        for i in range(agent_Base_Num):
            agent_Base_Loci = agent_Base_Loci_Full[i][len(agent_Base_Loci_Full[i]) - 1]
            if ((goal_X >= (agent_Base_Loci[0] - agent_Base_Loci[2] / 2)) and
                    (goal_X <= (agent_Base_Loci[0] + agent_Base_Loci[2] / 2)) and
                    (goal_Y >= (agent_Base_Loci[1] - agent_Base_Loci[3] / 2)) and
                    (goal_Y <= (agent_Base_Loci[1] + agent_Base_Loci[3] / 2))):
                in_Base_Flag = True
                base_Index = i
                break
        return in_Base_Flag, base_Index

    # This function intends to plot the sensed fire spot on the screen
    # Input: current screen, sensed fire spot list
    def sensed_Fire_Spot_Plot(self, screen, sensed_List, fire_Current_Map, current_Max_Intensity):
        # Search for all the sensing agents' data
        for i in range(len(sensed_List)):
            # Update the maximum intersity
            if current_Max_Intensity < fire_Current_Map[int(sensed_List[i][0])][int(sensed_List[i][1])]:
                current_Max_Intensity = fire_Current_Map[int(sensed_List[i][0])][int(sensed_List[i][1])]
            # Plot the fire spot using the red color the corresponds to the intensity
            pygame.draw.circle(screen, (
            fire_Current_Map[int(sensed_List[i][0])][int(sensed_List[i][1])] * 155 / current_Max_Intensity + 100, 0, 0),
                               (int(sensed_List[i][0]), int(sensed_List[i][1])), 1)

        return  current_Max_Intensity

    # Compute score for online and offline display
    def score_Calculation(self, fire_map_len, onfire_List, sensed_list, pruning_list, target_onFire_List, target_onFire_Flag, facility_penalty, environment_para, set_loci, time):
        time = time / 1000

        sensed_List_copy = sensed_list.copy()
        if len(sensed_List_copy) > 0:
            sensed_List_copy[0:0] = list(onfire_List).copy()
            modified_sensed_num = len(sensed_list) - len(np.unique(np.array(sensed_List_copy), axis=0).tolist()) + len(onfire_List)
        else:
            modified_sensed_num = 0

        if (len(onfire_List) + len(pruning_list)) > 0:
            overall_pruning_score = round(len(pruning_list) / (len(onfire_List) + len(pruning_list)) * 100.00, 2)
            preception_score = round((modified_sensed_num + len(pruning_list)) / (len(onfire_List) + len(pruning_list)) * 100.00, 2)
        else:
            overall_pruning_score = 0
            preception_score = 0

        if (modified_sensed_num + len(pruning_list)) > 0:
            action_score = round(len(pruning_list) / (modified_sensed_num + len(pruning_list)) * 100.00, 2)
        else:
            action_score = 0

        target_Num = 0
        target_onFire_Num = 0
        total_Negative_Score = 0
        for i in range(len(target_onFire_Flag)):
            for j in range(len(target_onFire_Flag[i])):
                target_onFire_Num += target_onFire_Flag[i][j]
                target_Num += 1
                if i == 0:
                    total_Negative_Score += target_onFire_List[i][j][len(target_onFire_List[i][j]) - 1] * facility_penalty[0]
                elif i == 1:
                    total_Negative_Score += target_onFire_List[i][j][len(target_onFire_List[i][j]) - 1] * facility_penalty[1]
                elif i == 2:
                    total_Negative_Score += target_onFire_List[i][j][len(target_onFire_List[i][j]) - 1] * facility_penalty[2]
                else:
                    total_Negative_Score += target_onFire_List[i][j][len(target_onFire_List[i][j]) - 1] * facility_penalty[3]

        total_Negative_Score += len(onfire_List) * set_loci[1][6] * (time ** set_loci[1][5])

        if set_loci[1][9] == 0:
            time_len = max(time - set_loci[1][1], 0)
            total_Negative_Expect = (len(onfire_List) + len(pruning_list)) * set_loci[1][6] * (time_len ** set_loci[1][5])
        else:
            total_Negative_Expect = 0
            for i in range(environment_para[2]):
                time_len = max(time - set_loci[1][1][i], 0)
                total_Negative_Expect += (len(onfire_List) + len(pruning_list)) * set_loci[1][6] * (time_len ** set_loci[1][5])

        if total_Negative_Expect > 0:
            total_Negative_percent = round(total_Negative_Score / total_Negative_Expect * 100.00, 2)
        else:
            total_Negative_percent = 0

        safe_Num = target_Num - target_onFire_Num
        facility_perception_score = round(safe_Num / target_Num * 100.00, 2)

        total_Negative_Score = round(total_Negative_Score, 2)
        return overall_pruning_score, preception_score, action_score, safe_Num, facility_perception_score, total_Negative_Score, total_Negative_percent

    # Display scores on the side bar
    def score_display(self, screen, font, font_Score, font_Scorelist, pos, score_list):
        [overall_pruning_score, preception_score, action_score, safe_Num, facility_perception_score, total_Negative_Score, total_Negative_percent] = score_list
        (pos_x, pos_y) = pos

        pygame.draw.rect(screen, (0, 0, 0), Rect((pos_x - 10, pos_y - 10), (580, 240)), 2)
        interval = 40
        screen.blit(font_Score.render('Score: ', True, (0, 0, 0)), (pos_x, pos_y))
        screen.blit(font.render(str(-total_Negative_Score), True, (192, 0, 0)), (pos_x + 360, pos_y))

        pygame.draw.line(screen, (0, 0, 0), (pos_x, pos_y + interval), (pos_x + 560, pos_y + interval), 2)

        screen.blit(font_Scorelist.render('Overall Firefighting Score:  ', True, (0, 0, 0)), (pos_x, pos_y + 1.5 * interval))
        screen.blit(font.render(str(overall_pruning_score), True, (22, 163, 26)), (pos_x + 360, pos_y + 1.5 * interval))

        screen.blit(font_Scorelist.render('Perception Score: ', True, (0, 0, 0)), (pos_x, pos_y + 2.5 * interval))
        screen.blit(font.render(str(preception_score), True, (22, 163, 26)), (pos_x + 360, pos_y + 2.5 * interval))

        screen.blit(font_Scorelist.render('Action Score: ', True, (0, 0, 0)), (pos_x, pos_y + 3.5 * interval))
        screen.blit(font.render(str(action_score), True, (22, 163, 26)), (pos_x + 360, pos_y + 3.5 * interval))

        screen.blit(font_Scorelist.render('Safe Facilities: ', True, (0, 0, 0)), (pos_x, pos_y + 4.5 * interval))
        screen.blit(font.render(str(safe_Num), True, (22, 163, 26)), (pos_x + 360, pos_y + 4.5 * interval))

    # Plot the fire region in the preview page
    def fire_region_plot(self, screen, font, fire_list):
        for i in range(len(fire_list)):
            pygame.draw.rect(screen, (255, 0, 0), Rect((fire_list[i][0], fire_list[i][1]), (100, 100)), 2)
            screen.blit(font.render(str(fire_list[i][2]), False, (255, 0, 0)), (fire_list[i][0] + 30,
                                                                                fire_list[i][1] + 30))

    # Plot lakes and store their loci information
    def lake_plot(self, screen, lake_list, lake_Loci, current_Time):
        for i in range(len(lake_list)):
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][len(lake_list[i]) - 1][0], lake_list[i][len(lake_list[i]) - 1][1]), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][len(lake_list[i]) - 1][0] + 80, lake_list[i][len(lake_list[i]) - 1][1] - 80), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][len(lake_list[i]) - 1][0] + 20, lake_list[i][len(lake_list[i]) - 1][1] + 80), 100)
            if lake_Loci != None:
                lake_Loci[i].append(
                [lake_list[i][len(lake_list[i]) - 1][0], lake_list[i][len(lake_list[i]) - 1][1], 100, np.floor(current_Time / 100)])
        return lake_Loci

    # Plot lakes
    def lake_plot_static(self, screen, lake_list):
        for i in range(len(lake_list)):
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][0], lake_list[i][1]), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][0] + 80, lake_list[i][1] - 80), 100)
            pygame.draw.circle(screen, (0, 191, 255), (lake_list[i][0] + 20, lake_list[i][1] + 80), 100)

    # Plot roads that connect each target
    def road_plot(self, screen, target_Loci):
        for i in range(len(target_Loci)):
            for j in range(i, len(target_Loci)):
                pygame.draw.line(screen, (139, 69, 19), (target_Loci[i][0][0], target_Loci[i][0][1]),
                                 (target_Loci[j][0][0], target_Loci[j][0][1]), 5)

