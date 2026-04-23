import time
import sys
import numpy as np

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

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.5
            self.cmd.motor_cmd[i].tau = 0.0

        # VMC parameters
        self.Kz = 300.0
        self.Dz = 10.0
        self.max_tau = 30.0

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

    def low_state_handler(self, msg: LowState_):
        print(f"Low state received. IMU: {msg.imu_state}, motor[0] q: {msg.motor_state[0].q}")
        self.low_state = msg
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq
        self.state_received = True

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

    def foot_relative_z(self, q_leg):
        _, q2, q3 = q_leg
        return -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)

    def jacobian_z(self, q_leg):
        _, q2, q3 = q_leg
        dz_dq1 = 0.0
        dz_dq2 = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        dz_dq3 = self.L_CALF * np.sin(q2 + q3)
        return np.array([dz_dq1, dz_dq2, dz_dq3], dtype=float)

    def foot_z_velocity(self, q_leg, dq_leg):
        Jz = self.jacobian_z(q_leg)
        return Jz @ dq_leg

    def smooth_phase(self, t, T=1.):
        return np.tanh(max(t, 0.0) / T)

    def desired_foot_z(self, z0, t, rise_height=0.08, T=1.):
        s = self.smooth_phase(t, T=T)
        return z0 - s * rise_height

    def desired_foot_dz(self, t):
        return 0.0

    def vmc_leg_tau(self, q_leg, dq_leg, z_des, dz_des):
        z = self.foot_relative_z(q_leg)
        dz = self.foot_z_velocity(q_leg, dq_leg)

        Fz = self.Kz * (z_des - z) + self.Dz * (dz_des - dz)
        Jz = self.jacobian_z(q_leg)

        tau_leg = Jz * Fz
        tau_leg = np.clip(tau_leg, -self.max_tau, self.max_tau)
        return tau_leg, z, dz, Fz
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
        T_stand = 5.0
        print("Low state received.")
        self.fixed=False

        # Save initial foot z for each leg
        self.z0_legs = np.zeros(4)
        for leg_id in range(4):
            q_leg = self.get_leg_q(leg_id)
            self.z0_legs[leg_id] = self.foot_relative_z(q_leg)

        self.start_time = time.perf_counter()

        while True:
            step_start = time.perf_counter()
            t = step_start - self.start_time
            if t > T_stand:
                if not self.fixed:
                    self.stand_up_joint_pos = self.joint_q.copy()
                    self.fixed = True
                phase = np.tanh((t - T_stand) / 1.2)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * self.stand_down_joint_pos[i] + (
                        1 - phase) * self.stand_up_joint_pos[i]
                    self.cmd.motor_cmd[i].kp = 50.0
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = 3.5
                    self.cmd.motor_cmd[i].tau = 0.0
            else:
                rise_height = 0.6
                T=1.
                for leg_id in range(4):
                    q_leg = self.get_leg_q(leg_id)
                    dq_leg = self.get_leg_dq(leg_id)

                    z_des = self.desired_foot_z(self.z0_legs[leg_id], t, rise_height=rise_height, T=T)
                    dz_des = self.desired_foot_dz(t)

                    tau_leg, z, dz, Fz = self.vmc_leg_tau(q_leg, dq_leg, z_des, dz_des)
                    tau_hip = self.hip_pd_tau(q_leg, dq_leg, leg_id)
                    tau_vmc, z, dz, Fz = self.vmc_leg_tau(q_leg, dq_leg, z_des, dz_des)
                    tau_leg = tau_vmc + tau_hip
                    tau_leg = np.clip(tau_leg, -self.max_tau, self.max_tau)
                    idx = 3 * leg_id
                    for j in range(3):
                        self.cmd.motor_cmd[idx + j].q = 0.0
                        self.cmd.motor_cmd[idx + j].kp = 0.0
                        self.cmd.motor_cmd[idx + j].dq = 0.0
                        self.cmd.motor_cmd[idx + j].kd = 0.5
                        self.cmd.motor_cmd[idx + j].tau = float(tau_leg[j])
                    print(f"Leg {leg_id} tau: {tau_leg}")
            self.publish_command()

            time_until_next_step = 0.002 - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


input("Press enter to start")

if __name__ == "__main__":
    interface = None if len(sys.argv) < 2 else sys.argv[1]
    controller = Go2Controller(interface)
    controller.run()