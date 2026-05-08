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
        self.last_print_time = 0.0

        # VMC PD Gains
        self.Kx, self.Dx = 180.0, 10.0
        self.Ky, self.Dy = 250.0, 12.0
        self.Kz, self.Dz = 400.0, 25.0
        self.tau_limits = np.array([23.0, 23.0, 40.0] * 4)

        
        self.stand_down_joint_pos = np.array([0.04, 1.22, -2.44] * 4)
        self.stand_up_joint_pos_target = np.array([0.0, 0.67, -1.3] * 4)

        self.xyz_des_flat = None 
        self.start_time = None

        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_gyro = np.zeros(3)
        self.roll_ref = 0.0
        self.pitch_ref = 0.0

        self.K_roll_body = 25.0
        self.D_roll_body = 2.0

        self.K_pitch_body = 50.0
        self.D_pitch_body = 5.0
        self.foot_body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

        self.foot_body_ids = []
        for name in self.foot_body_names:
            bid = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                name
            )

            if bid == -1:
                raise RuntimeError(f"Foot body '{name}' not found in MuJoCo model.")

            self.foot_body_ids.append(bid)

        print("Foot body ids:", dict(zip(self.foot_body_names, self.foot_body_ids)))
        
    def print_mujoco_names_containing(self, text):
        print(f"\nMuJoCo names containing '{text}':")

        print("\nBodies:")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and text.lower() in name.lower():
                print(f"  body {i}: {name}")

        print("\nSites:")
        for i in range(self.mj_model.nsite):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name and text.lower() in name.lower():
                print(f"  site {i}: {name}")

        print("\nGeoms:")
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and text.lower() in name.lower():
                print(f"  geom {i}: {name}")
    def _extract_robot_geometry(self):
        thigh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_thigh")
        calf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        self.L_THIGH = np.linalg.norm(self.mj_model.body_pos[calf_id])
        foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot")
        self.L_CALF = np.linalg.norm(self.mj_model.body_pos[foot_id])

    def low_state_handler(self, msg: LowState_):
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq

        self.imu_quat[:] = msg.imu_state.quaternion
        self.imu_gyro[:] = msg.imu_state.gyroscope

        self.state_received = True

    def rpy_from_quat(self, q):

        w, x, y, z = q

        # Roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    def quat_to_rotmat(self, q):
        w, x, y, z = q

        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ])
    def get_level_statics(self):
        self.mj_data.qpos[0:3] = 0.0
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qpos[self.qpos_adr] = self.joint_q

        mujoco.mj_forward(self.mj_model, self.mj_data)

        com = self.compute_whole_body_com()

        foot_pos = []
        for bid in self.foot_body_ids:
            foot_pos.append(self.mj_data.xpos[bid].copy())

        return com, foot_pos
    def compute_whole_body_com(self):
        masses = self.mj_model.body_mass
        xpos = self.mj_data.xipos

        total_mass = np.sum(masses[1:])
        com = np.sum(xpos[1:] * masses[1:, None], axis=0) / total_mass

        return com
    def compute_fz_distribution(self, com, foot_pos, tau_roll_des=0.0, tau_pitch_des=0.0):
        mg = self.mass * self.g
        A = np.zeros((3, 4))

        levers = []

        for i in range(4):
            lever = foot_pos[i] - com
            levers.append(lever)

            x = lever[0]
            y = lever[1]

            A[0, i] = 1.0
            A[1, i] = y
            A[2, i] = -x

        b = np.array([mg, tau_roll_des, tau_pitch_des])

        fz = np.linalg.pinv(A) @ b

        if time.perf_counter() - getattr(self, "last_fz_debug_time", 0.0) > 2.0:
            self.last_fz_debug_time = time.perf_counter()

            print("COM:", com)
            print("levers x,y:")
            for name, lev in zip(["FR", "FL", "RR", "RL"], levers):
                print(f"  {name}: x={lev[0]:+.3f}, y={lev[1]:+.3f}")

            print("A =")
            print(A)
            print("b =", b)
            print("raw fz =", fz)

        min_fz = 0.03 * mg / 4.0
        max_fz = 0.70 * mg

        return np.clip(fz, min_fz, max_fz)
    def foot_relative_xyz(self, q_leg):
        q1, q2, q3 = q_leg

        # Sagittal-plane position before hip ab/ad rotation.
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        z0 = -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)

        # Hip ab/ad rotates the leg in the y-z plane.
        y = -z0 * np.sin(q1)
        z = z0 * np.cos(q1)

        return np.array([x, y, z])
    def jacobian_xyz(self, q_leg):
        """
        Analytical Jacobian of foot position:

            foot_velocity = J_xyz @ joint_velocity

        J_xyz has shape 3x3:

            [dx/dq1  dx/dq2  dx/dq3]
            [dy/dq1  dy/dq2  dy/dq3]
            [dz/dq1  dz/dq2  dz/dq3]

        Then VMC maps Cartesian foot force to joint torque using:

            tau = J.T @ F

        where:

            F = [Fx, Fy, Fz]
        """

        q1, q2, q3 = q_leg

        # Sagittal position.
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        z0 = -self.L_THIGH * np.cos(q2) - self.L_CALF * np.cos(q2 + q3)

        # Partial derivatives of x.
        dx_dq1 = 0.0
        dx_dq2 = self.L_THIGH * np.cos(q2) + self.L_CALF * np.cos(q2 + q3)
        dx_dq3 = self.L_CALF * np.cos(q2 + q3)

        # Partial derivatives of z0.
        dz0_dq2 = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        dz0_dq3 = self.L_CALF * np.sin(q2 + q3)

        # y = -z0 * sin(q1)
        dy_dq1 = -z0 * np.cos(q1)
        dy_dq2 = -dz0_dq2 * np.sin(q1)
        dy_dq3 = -dz0_dq3 * np.sin(q1)

        # z = z0 * cos(q1)
        dz_dq1 = -z0 * np.sin(q1)
        dz_dq2 = dz0_dq2 * np.cos(q1)
        dz_dq3 = dz0_dq3 * np.cos(q1)

        return np.array([
            [dx_dq1, dx_dq2, dx_dq3],
            [dy_dq1, dy_dq2, dy_dq3],
            [dz_dq1, dz_dq2, dz_dq3],
        ])

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
                if self.xyz_des_flat is None:
                    self.xyz_des_flat = [
                        self.foot_relative_xyz(self.stand_up_joint_pos_target[i*3:i*3+3])
                        for i in range(4)
                    ]
                    print("3D VMC + Gravity Compensation Active.")

                com, foot_pos = self.get_level_statics()
                roll, pitch, yaw = self.rpy_from_quat(self.imu_quat)

                roll_rate = self.imu_gyro[0]
                pitch_rate = self.imu_gyro[1]

                roll_err = self.roll_ref - roll
                pitch_err = self.pitch_ref - pitch

                tau_roll_des = self.K_roll_body * roll_err - self.D_roll_body * roll_rate
                tau_pitch_des = self.K_pitch_body * pitch_err - self.D_pitch_body * pitch_rate
                fz_dist = self.compute_fz_distribution(com, foot_pos, tau_roll_des, tau_pitch_des)
                leg_errors = []
                tau_cmd_all = np.zeros(12)
                R_body_to_world = self.quat_to_rotmat(self.imu_quat)
                for i in range(4):
                    idx = i * 3
                    q_leg, dq_leg = self.joint_q[idx:idx+3], self.joint_dq[idx:idx+3]
                    xyz, dxyz = self.foot_relative_xyz(q_leg), (self.jacobian_xyz(q_leg) @ dq_leg)
                    err_xyz = self.xyz_des_flat[i] - xyz
                    leg_errors.append(np.linalg.norm(err_xyz))
                    F = np.zeros(3)

                    F[0] = self.Kx * (self.xyz_des_flat[i][0] - xyz[0]) - self.Dx * dxyz[0]
                    F[1] = self.Ky * (self.xyz_des_flat[i][1] - xyz[1]) - self.Dy * dxyz[1]
                    F[2] = self.Kz * (self.xyz_des_flat[i][2] - xyz[2]) - self.Dz * dxyz[2]



                    # Gravity is vertical force but we need to rotate it to the body frame to apply it correctly in the VMC control law, otherwise when the robot is tilted, gravity would not be vertical in the world frame and the compensation would be wrong, by rotating it to the body frame, we ensure that gravity compensation is always applied in the correct direction relative to the robot's orientation
                    F_world_grav = np.array([0.0, 0.0, -fz_dist[i]])
                    F_body_grav = R_body_to_world.T @ F_world_grav
                    F += F_body_grav

                    tau_leg = self.jacobian_xyz(q_leg).T @ F
                    tau_cmd_all[idx:idx+3] = tau_leg
                if t - self.last_print_time >= 2.0:
                    self.last_print_time = t

                    mean_err = np.mean(leg_errors)
                    max_err = np.max(leg_errors)

                    print(
                    f"[{t:.1f}s] "
                    f"pitch={np.degrees(pitch):+.2f} deg, "
                    f"tau_pitch_des={tau_pitch_des:+.2f} Nm | "
                    f"fz=({fz_dist[0]:.1f}, {fz_dist[1]:.1f}, {fz_dist[2]:.1f}, {fz_dist[3]:.1f}) N | "
                    f"Foot error mean={mean_err*1000:.1f} mm, max={max_err*1000:.1f} mm | "
                    f"FR={leg_errors[0]*1000:.1f}, FL={leg_errors[1]*1000:.1f}, "
                    f"RR={leg_errors[2]*1000:.1f}, RL={leg_errors[3]*1000:.1f} mm | "
                    f"tau_cmd (Nm) | "
                    f"FR=({tau_cmd_all[0]:.1f}, {tau_cmd_all[1]:.1f}, {tau_cmd_all[2]:.1f}), "
                    f"FL=({tau_cmd_all[3]:.1f}, {tau_cmd_all[4]:.1f}, {tau_cmd_all[5]:.1f}), "
                    f"RR=({tau_cmd_all[6]:.1f}, {tau_cmd_all[7]:.1f}, {tau_cmd_all[8]:.1f}), "
                    f"RL=({tau_cmd_all[9]:.1f}, {tau_cmd_all[10]:.1f}, {tau_cmd_all[11]:.1f})"
                )
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