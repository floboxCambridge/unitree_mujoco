import time
import sys
import numpy as np
import mujoco

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class Go2Controller:
    def __init__(self, interface=None):
        if interface is None:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, interface)

        # Low-level joint state
        self.low_state = None
        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)
        self.state_received = False

        self.crc = CRC()

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        self.desired_height=np.array([0.3, 0.5, 0.4])
        self.counter=0
        self.time_up=10.
        self.time_down=10.

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.5
            self.cmd.motor_cmd[i].tau = 0.0

        # VMC parameters
        self.Kz = 1000.0
        self.Dz = 100.0
        self.max_tau = 40.0

        self.L_THIGH = 0.213
        self.L_CALF = 0.213

        self.hip_ref = np.array([0.0057, -0.0057, 0.0057, -0.0057], dtype=float)
        self.K_hip = 40.0
        self.D_hip = 2.0
        self.max_hip_tau = 25.0
        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],
                                        dtype=float)

        self.z0_legs = None
        self.start_time = None
        self.saved_vmc_ref = False

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq
        self.state_received = True
        print(f"Low state received. motor q: {self.joint_q}")

    def hip_pd_tau(self, q_leg, dq_leg, leg_id):
        q1 = q_leg[0]
        dq1 = dq_leg[0]

        q1_des = self.hip_ref[leg_id]
        dq1_des = 0.0

        tau1 = self.K_hip * (q1_des - q1) + self.D_hip * (dq1_des - dq1)
        tau1 = np.clip(tau1, -self.max_hip_tau, self.max_hip_tau)

        return np.array([tau1, 0.0, 0.0], dtype=float)

    def get_leg_q(self, leg_id):
        idx = 3 * leg_id
        return self.joint_q[idx:idx+3].copy()

    def get_leg_dq(self, leg_id):
        idx = 3 * leg_id
        return self.joint_dq[idx:idx+3].copy()

    def foot_relative_xz(self, q_leg):
        _, q2, q3 = q_leg
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        z = -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)
        return np.array([x, z], dtype=float)


    def jacobian_xz(self, q_leg):
        _, q2, q3 = q_leg
        dx_dq1 = 0.0
        dx_dq2 = self.L_THIGH * np.cos(q2) + self.L_CALF * np.cos(q2 + q3)
        dx_dq3 = self.L_CALF * np.cos(q2 + q3)
        dz_dq1 = 0.0
        dz_dq2 = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        dz_dq3 = self.L_CALF * np.sin(q2 + q3)
        Jxz = np.array([
        [0.0, dx_dq2, dx_dq3],
        [0.0, dz_dq2, dz_dq3],
        ], dtype=float)

        return Jxz

    def foot_xz_velocity(self, q_leg, dq_leg):
        return self.jacobian_xz(q_leg) @ dq_leg


    def desired_foot_dz(self, t):
        return 0.0

    def vmc_leg_tau_2d(self, q_leg, dq_leg, xz_des, dxz_des):
        xz = self.foot_relative_xz(q_leg)
        dxz = self.foot_xz_velocity(q_leg, dq_leg)

        K = np.diag([120.0, self.Kz])
        D = np.diag([6.0, self.Dz])
        print(f"xz_des: {xz_des}, xz: {xz}")
        F = K @ (xz_des - xz) + D @ (dxz_des - dxz)

        Jxz = self.jacobian_xz(q_leg)

        tau_leg = Jxz.T @ F
        print(f"tau_leg before clipping: {tau_leg}")
        tau_leg = np.clip(tau_leg, -self.max_tau, self.max_tau)
        print(f"tau_leg after clipping: {tau_leg}")

        return tau_leg, xz, dxz, F
    def set_damping_mode(self, kd_hip=1.0, kd_leg=3.0):
        for leg_id in range(4):
            idx = 3 * leg_id

            # hip
            self.cmd.motor_cmd[idx + 0].q = 0.0
            self.cmd.motor_cmd[idx + 0].kp = 0.0
            self.cmd.motor_cmd[idx + 0].dq = 0.0
            self.cmd.motor_cmd[idx + 0].kd = kd_hip
            self.cmd.motor_cmd[idx + 0].tau = 0.0

            # thigh
            self.cmd.motor_cmd[idx + 1].q = 0.0
            self.cmd.motor_cmd[idx + 1].kp = 0.0
            self.cmd.motor_cmd[idx + 1].dq = 0.0
            self.cmd.motor_cmd[idx + 1].kd = kd_leg
            self.cmd.motor_cmd[idx + 1].tau = 0.0

            # calf
            self.cmd.motor_cmd[idx + 2].q = 0.0
            self.cmd.motor_cmd[idx + 2].kp = 0.0
            self.cmd.motor_cmd[idx + 2].dq = 0.0
            self.cmd.motor_cmd[idx + 2].kd = kd_leg
            self.cmd.motor_cmd[idx + 2].tau = 0.0
    def publish_command(self):
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def run(self):
        print("Waiting for low state...")
        while not self.state_received:
            time.sleep(0.01)
        T_stand = 100.0
        print("Low state received.")
        self.fixed=False
        self.start_time = time.perf_counter()
        

        while True:
            step_start = time.perf_counter()
            t = step_start - self.start_time

            if t > self.time_up and t < self.time_up + self.time_down:
                if not self.fixed:
                    self.stand_up_joint_pos = self.joint_q.copy()
                    self.fixed = True
                phase = np.tanh((t - self.time_up) / 2.)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * self.stand_down_joint_pos[i] + (
                        1 - phase) * self.stand_up_joint_pos[i]
                    self.cmd.motor_cmd[i].kp = 50.0
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = 3.5
                    self.cmd.motor_cmd[i].tau = 0.0
            elif t < self.time_up:
                rise_height = self.desired_height[self.counter]
                T = 1.0

                if not self.saved_vmc_ref:
                    self.xz0_legs = np.zeros((4, 2))

                    for leg_id in range(4):
                        q_leg = self.get_leg_q(leg_id)
                        self.xz0_legs[leg_id] = self.foot_relative_xz(q_leg)

                    self.saved_vmc_ref = True
                for leg_id in range(4):
                    q_leg = self.get_leg_q(leg_id)
                    dq_leg = self.get_leg_dq(leg_id)

                    xz0 = self.xz0_legs[leg_id]


                    x_des = xz0[0]
                    z_des = -  rise_height

                    xz_des = np.array([x_des, z_des], dtype=float)
                    dxz_des = np.array([0.0, 0.0], dtype=float)

                    tau_hip = self.hip_pd_tau(q_leg, dq_leg, leg_id)
                    tau_vmc, xz, dxz, F = self.vmc_leg_tau_2d(q_leg, dq_leg, xz_des, dxz_des)

                    tau_leg = tau_vmc + tau_hip
                    tau_leg = np.clip(tau_leg, -self.max_tau, self.max_tau)

                    idx = 3 * leg_id

                    tau_leg = tau_vmc + tau_hip 
                    tau_leg = np.clip(tau_leg, -self.max_tau, self.max_tau)
                    for j in range(3):
                        self.cmd.motor_cmd[idx + j].q = 0.0
                        self.cmd.motor_cmd[idx + j].kp = 0.0
                        self.cmd.motor_cmd[idx + j].dq = 0.0
                        self.cmd.motor_cmd[idx + j].kd = 0.5
                        self.cmd.motor_cmd[idx + j].tau = float(tau_leg[j])
            else:
                self.counter += 1
                self.fixed = False
                self.saved_vmc_ref = False
                self.start_time = time.perf_counter()

                if self.counter >= len(self.desired_height):
                    self.set_damping_mode()
                    self.publish_command()
                    exit(0)

            self.publish_command()

            time_until_next_step = 0.002 - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


input("Press enter to start")

if __name__ == "__main__":
    interface = None if len(sys.argv) < 2 else sys.argv[1]
    controller = Go2Controller(interface)
    controller.run()