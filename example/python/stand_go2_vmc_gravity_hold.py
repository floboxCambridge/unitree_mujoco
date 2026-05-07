import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simulate_python"))

from _unitree_sdk_path import ensure_unitree_sdk2py

ensure_unitree_sdk2py()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class Go2VmcGravityHoldController:
    def __init__(self, interface=None):
        if interface is None:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, interface)

        self.dt = 0.002
        self.stand_up_time = 5.0

        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.low_state = None
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
        self.init_motor_cmd()

        self.model_path = Path(__file__).resolve().parents[2] / "unitree_robots/go2/go2.xml"
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        self.load_mujoco_model()

        self.g = 9.81
        self.Fz_per_leg = self.mass * self.g / 4.0

        self.L_THIGH = 0.213
        self.L_CALF = 0.213

        self.Kx = 160.0
        self.Dx = 8.0
        self.Kz = 800.0
        self.Dz = 55.0
        self.gravity_comp_scale = 1.0

        self.hip_ref = np.array([0.0057, -0.0057, 0.0057, -0.0057], dtype=float)
        self.K_hip = 40.0
        self.D_hip = 2.0
        self.max_hip_tau = 18.0
        self.tau_limits = np.array([23.0, 23.0, 40.0] * 4, dtype=float)

        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375,
            -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375,
            -0.0473455, 1.22187, -2.44375,
        ], dtype=float)
        self.stand_up_joint_pos = np.array([
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
        ], dtype=float)

        self.foot_xz_ref = None
        self.start_time = None

    def init_motor_cmd(self):
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.5
            self.cmd.motor_cmd[i].tau = 0.0

    def load_mujoco_model(self):
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.mj_data = mujoco.MjData(self.mj_model)
            joint_ids = [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ]
            self.qpos_adr = np.array([self.mj_model.jnt_qposadr[jid] for jid in joint_ids])
            self.dof_adr = np.array([self.mj_model.jnt_dofadr[jid] for jid in joint_ids])
            self.mass = float(np.sum(self.mj_model.body_mass))
            self.use_mujoco = True
            print(f"Loaded MuJoCo model: {self.model_path}")
        except Exception as exc:
            print(f"Gravity compensation disabled: failed to load MuJoCo model: {exc}")
            self.mj_model = None
            self.mj_data = None
            self.qpos_adr = np.array([], dtype=int)
            self.dof_adr = np.array([], dtype=int)
            self.mass = 15.0
            self.use_mujoco = False

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq

        base_quat = np.array(msg.imu_state.quaternion, dtype=float)
        quat_norm = np.linalg.norm(base_quat)
        if quat_norm > 1e-6:
            self.base_quat = base_quat / quat_norm

        self.state_received = True

    def get_leg_q(self, leg_id):
        idx = 3 * leg_id
        return self.joint_q[idx:idx + 3].copy()

    def get_leg_dq(self, leg_id):
        idx = 3 * leg_id
        return self.joint_dq[idx:idx + 3].copy()

    def foot_relative_xz(self, q_leg):
        _, q2, q3 = q_leg
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        z = -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)
        return np.array([x, z], dtype=float)

    def jacobian_xz(self, q_leg):
        _, q2, q3 = q_leg
        dx_dq2 = self.L_THIGH * np.cos(q2) + self.L_CALF * np.cos(q2 + q3)
        dx_dq3 = self.L_CALF * np.cos(q2 + q3)
        dz_dq2 = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        dz_dq3 = self.L_CALF * np.sin(q2 + q3)
        return np.array([
            [0.0, dx_dq2, dx_dq3],
            [0.0, dz_dq2, dz_dq3],
        ], dtype=float)

    def foot_xz_velocity(self, q_leg, dq_leg):
        return self.jacobian_xz(q_leg) @ dq_leg

    def hip_pd_tau(self, q_leg, dq_leg, leg_id):
        tau = self.K_hip * (self.hip_ref[leg_id] - q_leg[0]) - self.D_hip * dq_leg[0]
        tau = np.clip(tau, -self.max_hip_tau, self.max_hip_tau)
        return np.array([tau, 0.0, 0.0], dtype=float)

    def vmc_leg_tau(self, q_leg, dq_leg, xz_des):
        xz = self.foot_relative_xz(q_leg)
        dxz = self.foot_xz_velocity(q_leg, dq_leg)

        stiffness = np.diag([self.Kx, self.Kz])
        damping = np.diag([self.Dx, self.Dz])
        force = stiffness @ (xz_des - xz) - damping @ dxz
        force[1] -= self.Fz_per_leg

        return self.jacobian_xz(q_leg).T @ force

    def gravity_compensation_torques(self):
        if not self.use_mujoco:
            return np.zeros(12)

        self.mj_data.qpos[0:3] = 0.0
        self.mj_data.qpos[3:7] = self.base_quat
        self.mj_data.qpos[self.qpos_adr] = self.joint_q
        self.mj_data.qvel[:] = 0.0

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data.qfrc_bias[self.dof_adr].copy()

    def publish_command(self):
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def set_joint_stand_up_command(self, t):
        phase = np.tanh(t / 1.2)
        for i in range(12):
            self.cmd.motor_cmd[i].q = (
                phase * self.stand_up_joint_pos[i]
                + (1.0 - phase) * self.stand_down_joint_pos[i]
            )
            self.cmd.motor_cmd[i].kp = phase * 50.0 + (1.0 - phase) * 20.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

    def save_vmc_reference(self):
        self.foot_xz_ref = np.zeros((4, 2), dtype=float)
        for leg_id in range(4):
            self.foot_xz_ref[leg_id] = self.foot_relative_xz(self.get_leg_q(leg_id))
        print(f"VMC hold enabled. Foot x/z references: {self.foot_xz_ref}")

    def set_vmc_hold_command(self):
        if self.foot_xz_ref is None:
            self.save_vmc_reference()

        tau_cmd = np.zeros(12, dtype=float)
        tau_gravity = self.gravity_compensation_torques()

        for leg_id in range(4):
            idx = 3 * leg_id
            q_leg = self.get_leg_q(leg_id)
            dq_leg = self.get_leg_dq(leg_id)
            tau_cmd[idx:idx + 3] = (
                self.vmc_leg_tau(q_leg, dq_leg, self.foot_xz_ref[leg_id])
                + self.hip_pd_tau(q_leg, dq_leg, leg_id)
                + self.gravity_comp_scale * tau_gravity[idx:idx + 3]
            )

        tau_cmd = np.clip(tau_cmd, -self.tau_limits, self.tau_limits)

        for i in range(12):
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.8
            self.cmd.motor_cmd[i].tau = float(tau_cmd[i])

    def run(self):
        print("Waiting for low state...")
        while not self.state_received:
            time.sleep(0.01)

        print("Low state received.")
        self.start_time = time.perf_counter()

        while True:
            step_start = time.perf_counter()
            t = step_start - self.start_time

            if t < self.stand_up_time:
                self.set_joint_stand_up_command(t)
            else:
                self.set_vmc_hold_command()

            self.publish_command()

            sleep_time = self.dt - (time.perf_counter() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


try:
    input("Press enter to start")
except EOFError:
    print("No interactive stdin available; starting immediately.")

if __name__ == "__main__":
    interface_name = None if len(sys.argv) < 2 else sys.argv[1]
    controller = Go2VmcGravityHoldController(interface_name)
    controller.run()
