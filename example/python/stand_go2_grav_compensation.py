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


class Go2Controller:
    def __init__(self, interface=None):
        if interface is None:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, interface)
            
        self.model_path = Path(__file__).resolve().parents[2] / "unitree_robots/go2/go2.xml"
        self.unitree_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.mj_data = mujoco.MjData(self.mj_model)
            
            self._extract_robot_geometry()
            self.mass = float(np.sum(self.mj_model.body_mass))
            
            self.joint_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.unitree_joint_names]
            self.qpos_adr = np.array([self.mj_model.jnt_qposadr[jid] for jid in self.joint_ids])
            
            print(f"Robot Info: Mass={self.mass:.2f}kg, Thigh={self.L_THIGH:.3f}m, Calf={self.L_CALF:.3f}m")
        except Exception as e:
            print(f"Critical MuJoCo Load Error: {e}")
            sys.exit(1)

        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)
        self.state_received = False

        self.crc = CRC()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)

        self.g = 9.81
        self.dt = 0.002
        self.time_up = 5.0
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.level_flag = 0xFF

        # VMC PD Gains
        self.Kx, self.Dx = 180.0, 10.0
        self.Kz, self.Dz = 1000.0, 70.0  
        self.tau_limits = np.array([23.0, 23.0, 40.0] * 4)

        self.hip_ref = np.array([0.005, -0.005, 0.005, -0.005])
        self.K_hip, self.D_hip = 40.0, 2.0
        
        self.stand_down_joint_pos = np.array([0.04, 1.22, -2.44] * 4)
        self.stand_up_joint_pos_target = np.array([0.0, 0.67, -1.3] * 4)

        self.xz_des_flat = None 
        self.start_time = None

    def _extract_robot_geometry(self):
        thigh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_thigh")
        calf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        self.L_THIGH = np.linalg.norm(self.mj_model.body_pos[calf_id])
        foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot")
        self.L_CALF = 0.213

    def low_state_handler(self, msg: LowState_):
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq
        self.state_received = True

    def get_level_statics(self):
        self.mj_data.qpos[0:3] = 0.0
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qpos[self.qpos_adr] = self.joint_q
        mujoco.mj_forward(self.mj_model, self.mj_data)

        com = self.mj_data.subtree_com[1].copy()

        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        foot_pos = []
        for n in foot_names:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
            foot_pos.append(self.mj_data.site_xpos[sid].copy())

        return com, foot_pos

    def compute_fz_distribution(self, com, foot_pos):
        mg = self.mass * self.g
        A = np.zeros((3, 4))

        for i in range(4):
            lever = foot_pos[i] - com

            A[0, i] = 1.0
            A[1, i] = lever[0]
            A[2, i] = lever[1]

        b = np.array([mg, 0.0, 0.0])

        fz = np.linalg.pinv(A) @ b

        # Avoid negative vertical forces
        return np.clip(fz, 0.05 * mg / 4.0, mg)
    def foot_relative_xz(self, q_leg):
        _, q2, q3 = q_leg
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        z = -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)
        return np.array([x, z])
    def jacobian_xz(self, q_leg):
        _, q2, q3 = q_leg
        dx_dq2 = self.L_THIGH * np.cos(q2) + self.L_CALF * np.cos(q2 + q3)
        dx_dq3 = self.L_CALF * np.cos(q2 + q3)
        dz_dq2 = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        dz_dq3 = self.L_CALF * np.sin(q2 + q3)
        return np.array([[0.0, dx_dq2, dx_dq3], [0.0, dz_dq2, dz_dq3]])

    def run(self):
        print("Waiting for state...")
        while not self.state_received: time.sleep(0.01)
        self.start_time = time.perf_counter()

        while True:
            t = time.perf_counter() - self.start_time
            
            if t < self.time_up:
                phase = np.tanh(t / 1.5)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * self.stand_up_joint_pos_target[i] + (1-phase) * self.stand_down_joint_pos[i]
                    self.cmd.motor_cmd[i].kp, self.cmd.motor_cmd[i].kd = (phase*60 + 20), 3.5
                    self.cmd.motor_cmd[i].tau = 0.0
            else:
                if self.xz_des_flat is None:
                    self.xz_des_flat = [self.foot_relative_xz(self.stand_up_joint_pos_target[i*3:i*3+3]) for i in range(4)]
                    print("Gravity Compensation Active (Flat Stance).")

                com, foot_pos = self.get_level_statics()
                fz_dist = self.compute_fz_distribution(com, foot_pos)
                
                tau_cmd_all = np.zeros(12)
                for i in range(4):
                    idx = i * 3
                    q_leg, dq_leg = self.joint_q[idx:idx+3], self.joint_dq[idx:idx+3]
                    xz, dxz = self.foot_relative_xz(q_leg), (self.jacobian_xz(q_leg) @ dq_leg)
                    
                    F = np.zeros(2)

                    F[0] = self.Kx * (self.xz_des_flat[i][0] - xz[0]) - self.Dx * dxz[0]
                    F[1] = self.Kz * (self.xz_des_flat[i][1] - xz[1]) - self.Dz * dxz[1]

                    # Gravity support only on vertical axis
                    F[1] -= fz_dist[i]

                    tau_leg = self.jacobian_xz(q_leg).T @ F

                    # Hip PD only on hip joint
                    tau_hip = self.K_hip * (self.hip_ref[i] - q_leg[0]) - self.D_hip * dq_leg[0]
                    tau_leg[0] += tau_hip

                    tau_cmd_all[idx:idx+3] = tau_leg

                tau_cmd_all = np.clip(tau_cmd_all, -self.tau_limits, self.tau_limits)
                for i in range(12):
                    self.cmd.motor_cmd[i].q, self.cmd.motor_cmd[i].kp = 0.0, 0.0
                    self.cmd.motor_cmd[i].dq, self.cmd.motor_cmd[i].kd = 0.0, 1.2
                    self.cmd.motor_cmd[i].tau = float(tau_cmd_all[i])

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            
            elapsed = time.perf_counter() - (t + self.start_time)
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

if __name__ == "__main__":
    controller = Go2Controller()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nShutdown.")