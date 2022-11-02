# 1D discrete model
# Concentration of an attractant only as a function of x, c(x)
# Discrete time, t, with a fixed timestep, dt.
# A fixed velocity

import numpy as np
import matplotlib.pyplot as plt


class Chemotaxis:
    def __init__(self, velocity=1, dt=1, direction=1, x=0, s=0, maxA=1, k=1, cType="flat"):
        self.velocity = velocity  # moving 1 block per time instance
        self.dt = dt  # unit time

        self.trajectory = []
        self.direction = direction  # initial direction
        self.init_direction = direction
        self.x = x  # initial location
        self.init_x = x
        self.xPre = None
        self.s = s  # initial condition of s
        self.init_s = s
        self.cType = cType

        # setting parameter arbitrarily
        self.maxA = maxA
        self.k = k

        self.minA = None
        self.tau = None

    def d_concentration(self,):
        if self.cType == "flat":
            return 0
        elif self.cType == "linear":
            try:
                return self.x - self.xPre
            except TypeError:
                return 0

    def update_s(self):
        # Euler integration
        self.s = self.s + self.dt * (- self.s + self.d_concentration()) / self.tau

    def p_tumble(self):
        return self.minA + (self.maxA - self.minA) / (1 + np.exp(self.k * self.s))

    def take_one_step(self):
        self.xPre = self.x
        p = self.p_tumble()
        self.direction = self.direction * np.random.choice([-1, 1], p=[p, 1 - p])
        self.x = self.x + self.direction * self.velocity * self.dt

    def run_simulation(self, minA, tau, no_steps=100):
        self.minA = minA
        self.tau = tau
        self.x = self.init_x
        self.trajectory = [self.x]
        self.xPre = None
        self.s = self.init_s
        self.direction = self.init_direction
        self.tumble_ps = []
        self.s_list = [self.s]
        times = [x for x in range(no_steps + 1)]
        condition = f"A_min={self.minA:.3f}, tau_s={self.tau:.3f}"
        for n in range(no_steps):
            self.update_s()
            self.s_list.append(self.s)
            self.tumble_ps.append(self.p_tumble())
            self.take_one_step()
            self.trajectory.append(self.x)
        plt.plot(times, self.trajectory, label=condition)
        plt.ylabel("discrete x position")
        plt.xlabel("time t")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def plt_multiple_traj(self, N, minA, tau, no_steps=100):
        trajectories = []
        conditions = []
        times = [x for x in range(no_steps + 1)]
        for i in range(N):
            self.minA = minA[i]
            self.tau = tau[i]
            self.x = self.init_x
            self.trajectory = [self.x]
            self.xPre = None
            self.direction = self.init_direction
            self.s = self.init_s
            for n in range(no_steps):
                self.update_s()
                self.take_one_step()
                self.trajectory.append(self.x)
            trajectories.append(self.trajectory)
            conditions.append(f"A_min={self.minA:.3f}, tau_s={self.tau:.3f}")

        for t, c in zip(trajectories, conditions):
            plt.plot(times, t, label=c)
            plt.ylabel("discrete x position")
            plt.xlabel("time t")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
