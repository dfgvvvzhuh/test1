import math
import random

import numpy as np


class UnloadingEnvironment(object):
    height = ground_length = ground_width = 100
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6
    p_noisy_los = 10 ** (-13)
    p_noisy_nlos = 10 ** (-11)
    flight_speed = 50.  
    # f_ue = 6e8  
    f_ue = 2e8  
    f_uav = 1.2e9  
    r = 10 ** (-27)  
    s = 1000 
    p_uplink = 0.1 
    alpha0 = 1e-5  
    T = 320 
    t_fly = 1
    t_com = 7
    delta_t = t_fly + t_com  
    v_ue = 1   
    slot_num = int(T / delta_t) 
    m_uav = 9.65 
    uav_number = 2
    sum_task_size = [100 * 1048576] * uav_number
    loc_uav = np.random.randint(50, 51, size=[uav_number, 2])
    e_battery_uav = [500000] * uav_number  
    M = 4
    block_flag_list = np.random.randint(0, 2, M)
    loc_ue_list = np.random.randint(0, 101, size=[M, 2]) 
    task_list = np.random.randint(2097153, 2621440, M)
    action_bound = [-1, 1]
    action_dim = 4
    state_dim = uav_number * 4 + M * 4 

    def __init__(self):
        self.state = np.zeros((self.uav_number, self.state_dim))
        for agent in range(self.uav_number):
            self.start_state = np.append(self.e_battery_uav, np.ravel(self.loc_uav))
            self.start_state = np.append(self.start_state, self.sum_task_size)
            self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
            self.start_state = np.append(self.start_state, self.task_list)
            self.start_state = np.append(self.start_state, self.block_flag_list)
            self.state[agent] = self.start_state


    def reset_env(self):
        self.sum_task_size = [100 * 1048576] * self.uav_number
        self.e_battery_uav = [500000] * self.uav_number
        self.loc_uav = np.random.randint(50, 51, size=[self.uav_number, 2])
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
        self.reset_step()

    def reset_step(self):
        self.task_list = np.random.randint(2621440, 3145729, self.M)
        self.block_flag_list = np.random.randint(0, 2, self.M)

    def reset(self):
        self.reset_env()
        for agent in range(self.uav_number):
            self.start_state = np.append(self.e_battery_uav, np.ravel(self.loc_uav))
            self.start_state = np.append(self.start_state, self.sum_task_size)
            self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
            self.start_state = np.append(self.start_state, self.task_list)
            self.start_state = np.append(self.start_state, self.block_flag_list)
            self.state[agent] = self.start_state
        return self._get_obs()

    def _get_obs(self):
        for agent in range(self.uav_number):
            self.start_state = np.append(self.e_battery_uav, np.ravel(self.loc_uav))
            self.start_state = np.append(self.start_state, self.sum_task_size)
            self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
            self.start_state = np.append(self.start_state, self.task_list)
            self.start_state = np.append(self.start_state, self.block_flag_list)
            self.state[agent] = self.start_state
        return self.state

    def step(self, action):
        step_redo = [False] * self.uav_number
        is_terminal = [False] * self.uav_number
        offloading_ratio_change = [False] * self.uav_number
        reset_dist = [False] * self.uav_number
        rewards = np.ones((self.uav_number, 1))
        rewards_result = []
        for agent in range(self.uav_number):
            actionT = (action[agent] + 1) / 2
            if actionT[0] == 1:
                ue_id = self.M - 1
            else:
                ue_id = int(self.M * actionT[0])
            theta = actionT[1] * np.pi * 2
            offloading_ratio = actionT[3]
            task_size = self.task_list[ue_id]
            block_flag = self.block_flag_list[ue_id]
            dis_fly = actionT[2] * self.flight_speed * self.t_fly
            e_fly = (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5
            dx_uav = dis_fly * math.cos(theta)
            dy_uav = dis_fly * math.sin(theta)
            loc_uav_after_fly_x = self.loc_uav[agent][0] + dx_uav
            loc_uav_after_fly_y = self.loc_uav[agent][1] + dy_uav
            t_server = offloading_ratio * task_size / (self.f_uav / self.s)
            e_server = self.r * self.f_uav ** 3 * t_server

            if self.sum_task_size[agent] == 0:
                is_terminal[agent] = True
                reward = 0
            elif self.sum_task_size[agent] - self.task_list[ue_id] < 0:
                self.task_list = np.ones(self.M) * self.sum_task_size[agent]
                reward = 0
                step_redo[agent] = True
            elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length:
                reset_dist[agent] = True
                delay = self.com_delay(self.loc_ue_list[ue_id], self.loc_uav[agent], offloading_ratio, task_size, block_flag, agent) 
                reward = -delay
                self.e_battery_uav[agent] = self.e_battery_uav[agent] - e_server
                self.reset2(delay, self.loc_uav[agent][0], self.loc_uav[agent][1], offloading_ratio, task_size, ue_id, agent)
            elif self.e_battery_uav[agent] < e_fly or self.e_battery_uav[agent] - e_fly < e_server:
                delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                       0, task_size, block_flag, agent) 
                reward = -delay
                self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, 0, task_size, ue_id, agent)
                offloading_ratio_change[agent] = True
                delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                       offloading_ratio, task_size, block_flag, agent)
                reward = -delay
                self.e_battery_uav[agent] = self.e_battery_uav[agent] - e_fly - e_server
                self.loc_uav[agent][0] = loc_uav_after_fly_x 
                self.loc_uav[agent][1] = loc_uav_after_fly_y
                self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, offloading_ratio, task_size,
                                               ue_id, agent)
            rewards[agent] = reward

        rewards = rewards.sum()
        rewards_result.append(rewards)
        rewards_result = rewards_result * self.uav_number

        return self._get_obs(), rewards_result, is_terminal, {} , step_redo, offloading_ratio_change, reset_dist

    def reset2(self, delay, x, y, offloading_ratio, task_size, ue_id, agent):
        self.sum_task_size[agent] -= self.task_list[ue_id]
        for i in range(self.M):
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2 
            dis_ue = tmp[1] * self.delta_t * self.v_ue
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.reset_step()

    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag, agent):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)
        t_tr = offloading_ratio * task_size / trans_rate
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return max([t_tr + t_edge_com, t_local_com])
