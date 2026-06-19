import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulate_python"))

from _unitree_sdk_path import ensure_unitree_sdk2py

ensure_unitree_sdk2py()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
PosStopF = 2.146e9
VelStopF = 16000.0

class Go2Controller:
    def __init__(self, interface=None, sim=False):
        if interface is None:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, interface)

        self.model_path = Path(__file__).resolve().parents[3] / "unitree_robots/go2/go2.xml"

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

            self.joint_ids = [
                mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name
                )
                for name in self.unitree_joint_names
            ]

            self.qpos_adr = np.array([
                self.mj_model.jnt_qposadr[jid]
                for jid in self.joint_ids
            ])

            print(
                f"Robot Info: Mass={self.mass:.2f}kg, "
                f"Thigh={self.L_THIGH:.3f}m, Calf={self.L_CALF:.3f}m"
            )

        except Exception as e:
            print(f"Critical MuJoCo Load Error: {e}")
            sys.exit(1)

        # ------------------------------------------------------------
        # Runtime state.
        # IMPORTANT: define all these before starting DDS subscribers.
        # The subscriber callback can run immediately after Init().
        # ------------------------------------------------------------
        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)
        self.state_received = False

        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_gyro = np.zeros(3)

        self.roll_ref = 0.0
        self.pitch_ref = 0.0

        self.start_joint_pos = np.zeros(12)
        self.vmc_joint_ref = np.zeros(12)
        self.vmc_initial_height = None
        self.damping_started = False
        self.start_time = None
        self.vmc_start_time = None
        self.last_print_time = 0.0

        # ------------------------------------------------------------
        # Control timing.
        # ------------------------------------------------------------
        self.g = 9.81
        self.dt = 0.002

        self.pose1_duration = 1.0   # match go2_stand_example targetPose 1
        self.pose2_duration = 1.0   # match go2_stand_example targetPose 2
        self.pose2_hold_time = 2.0  # match example hold at targetPose 2
        self.vmc_blend_time = 1.5
        self.vmc_exit_blend_time = 2.0
        self.vmc_height_transition_time = 3.0
        self.vmc_settle_hold_time = 5.0
        self.vmc_low_hold_time = 5.0
        self.vmc_high_hold_time = 5.0
        self.vmc_hold_time = (
            3.0 * self.vmc_height_transition_time
            + self.vmc_settle_hold_time
            + self.vmc_low_hold_time
            + self.vmc_high_hold_time
        )
        self.stand_down_duration = 2.0
        self.control_phase = "ramp"

        # ------------------------------------------------------------
        # Real robot conservative VMC gains.
        # ------------------------------------------------------------
        self.Kx, self.Dx = 80.0, 6.0
        self.Ky, self.Dy = 100.0, 6.0
        self.Kz, self.Dz = 220.0, 15.0

        self.K_roll_body = 10.0
        self.D_roll_body = 1.0

        self.K_pitch_body = 15.0
        self.D_pitch_body = 1.5

        self.tau_limits = np.array([16.0, 16.0, 28.0] * 4)
        self.vmc_joint_kp = 12.0
        self.vmc_joint_kd = 1.5
        self.vmc_tau_scale = 0.5
        self.vmc_target_height = 0.32
        self.vmc_low_height = 0.20
        self.vmc_high_height = 0.40
        self.exit_joint_kp = 35.0
        self.exit_joint_kd = 4.0

        # ------------------------------------------------------------
        # Joint references.
        # ------------------------------------------------------------
        self.target_pose_1 = np.array([
            0.0, 1.36, -2.65,
            0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65,
            0.2, 1.36, -2.65,
        ])
        self.target_pose_2 = np.array([
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
        ])
        self.stand_up_joint_pos_target = self.target_pose_2.copy()

        # ------------------------------------------------------------
        # Foot bodies for MuJoCo statics.
        # Order must match joint order: FR, FL, RR, RL.
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Height target.
        # For now: hold 20 cm only.
        # ------------------------------------------------------------
        self.height_sequence = [0.20]
        self.height_period = 10.0
        self.height_transition_time = 3.0

        self.xyz_nominal_flat = None

        # ------------------------------------------------------------
        # DDS low-level command setup.
        # ------------------------------------------------------------
        self.crc = CRC()

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.init_low_cmd()

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)

        # ------------------------------------------------------------
        # Release high-level motion mode once, after DDS is initialized.
        # ------------------------------------------------------------
        if not sim:
            self.release_motion_mode()
        else:
            print("Simulation mode: skipping MotionSwitcherClient/SportClient release.")
    def smoothstep(self, s):
        """
        Smooth interpolation from 0 to 1.

        s should be between 0 and 1.
        This avoids sudden jumps in desired height.
        """
        s = np.clip(s, 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)
    
    def init_low_cmd(self):
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = PosStopF
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = VelStopF
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
    def send_damping_command(self, kd=2.0):
        """
        Damping-only mode.

        This does not command a joint position.
        It only damps joint velocity.

        On the real robot this is much safer than abruptly sending zero torque,
        because the legs do not go completely passive instantly.
        """

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = PosStopF
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = VelStopF
            self.cmd.motor_cmd[i].kd = kd
            self.cmd.motor_cmd[i].tau = 0.0

    def release_motion_mode(self):
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()

        while result["name"]:
            print(f"Active motion mode: {result['name']} -> releasing")
            self.sc.StandDown()
            time.sleep(0.5)
            self.msc.ReleaseMode()
            time.sleep(0.5)
            status, result = self.msc.CheckMode()

        print("Motion mode released.")


    def send_zero_torque(self):
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = PosStopF
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = VelStopF
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
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
    def get_desired_height(self, t_vmc):
        """
        Return desired body height as a function of time.

        Heights cycle like this:

            0-10s    : 20 cm
            10-20s   : transition/stay toward 30 cm
            20-30s   : transition/stay toward 35 cm
            30-40s   : transition/stay toward 20 cm
            repeat...

        The transition is smoothed during the first few seconds of each 10s block.
        """

        heights = self.height_sequence
        n = len(heights)

        block = int(t_vmc // self.height_period)
        phase_time = t_vmc - block * self.height_period

        current_idx = block % n
        next_idx = (block + 1) % n

        h0 = heights[current_idx]
        h1 = heights[next_idx]

        # Only transition during the first self.height_transition_time seconds.
        s = phase_time / self.height_transition_time
        alpha = self.smoothstep(s)

        return (1.0 - alpha) * h0 + alpha * h1
    def get_desired_foot_targets(self, desired_height):
        """
        Build desired foot positions for all four legs.

        x and y come from the nominal stand-up pose.
        z is changed according to desired body height.

        Because the foot is below the body:

            foot_z_des = -desired_height
        """

        targets = []

        for i in range(4):
            target = self.xyz_nominal_flat[i].copy()
            target[2] = -desired_height
            targets.append(target)

        return targets

    def get_scheduled_vmc_height(self, t_vmc):
        transition = self.vmc_height_transition_time

        if t_vmc < transition:
            phase = self.smoothstep(t_vmc / transition)
            return (
                (1.0 - phase) * self.vmc_initial_height
                + phase * self.vmc_target_height
            )

        t_vmc -= transition
        if t_vmc < self.vmc_settle_hold_time:
            return self.vmc_target_height

        t_vmc -= self.vmc_settle_hold_time
        if t_vmc < transition:
            phase = self.smoothstep(t_vmc / transition)
            return (
                (1.0 - phase) * self.vmc_target_height
                + phase * self.vmc_low_height
            )

        t_vmc -= transition
        if t_vmc < self.vmc_low_hold_time:
            return self.vmc_low_height

        t_vmc -= self.vmc_low_hold_time
        if t_vmc < transition:
            phase = self.smoothstep(t_vmc / transition)
            return (
                (1.0 - phase) * self.vmc_low_height
                + phase * self.vmc_high_height
            )

        return self.vmc_high_height

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
        max_fz = 10* mg

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
        while not self.state_received:
            time.sleep(0.01)

        self.start_joint_pos[:] = self.joint_q.copy()
        print("Captured real start joint position:")
        print(self.start_joint_pos)

        self.start_time = time.perf_counter()
        pose1_end = self.pose1_duration
        pose2_end = pose1_end + self.pose2_duration
        settle_end = pose2_end + self.pose2_hold_time
        vmc_end = settle_end + self.vmc_hold_time
        stand_down_end = vmc_end + self.stand_down_duration

        while True:
            t = time.perf_counter() - self.start_time

            # -------------------------
            # Phase 1: example pose initialization.
            # -------------------------
            if t < pose1_end:
                phase = self.smoothstep(t / self.pose1_duration)
                q_des = (1.0 - phase) * self.start_joint_pos + phase * self.target_pose_1

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = float(q_des[i])
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 60.0
                    self.cmd.motor_cmd[i].kd = 5.0
                    self.cmd.motor_cmd[i].tau = 0.0

                if t - self.last_print_time >= 1.0:
                    self.last_print_time = t
                    print(f"[{t:.1f}s] moving to targetPose 1, phase={phase:.2f}")

            elif t < pose2_end:
                phase = self.smoothstep((t - pose1_end) / self.pose2_duration)
                q_des = (1.0 - phase) * self.target_pose_1 + phase * self.target_pose_2

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = float(q_des[i])
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 60.0
                    self.cmd.motor_cmd[i].kd = 5.0
                    self.cmd.motor_cmd[i].tau = 0.0

                if t - self.last_print_time >= 1.0:
                    self.last_print_time = t
                    print(f"[{t:.1f}s] moving to targetPose 2, phase={phase:.2f}")

            elif t < settle_end:
                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = float(self.target_pose_2[i])
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 60.0
                    self.cmd.motor_cmd[i].kd = 5.0
                    self.cmd.motor_cmd[i].tau = 0.0

                if t - self.last_print_time >= 1.0:
                    self.last_print_time = t
                    print(f"[{t:.1f}s] holding targetPose 2 before VMC")

            # -------------------------
            # Phase 2: VMC hold.
            # -------------------------
            
            elif t < vmc_end:
                if self.xyz_nominal_flat is None:
                    self.xyz_nominal_flat = [
                        self.foot_relative_xyz(self.joint_q[i*3:i*3+3])
                        for i in range(4)
                    ]
                    self.vmc_initial_height = float(
                        np.mean([-xyz[2] for xyz in self.xyz_nominal_flat])
                    )
                    self.vmc_joint_ref[:] = self.joint_q.copy()

                    self.vmc_start_time = t

                    print("3D VMC active.")
                    print(
                        "Blending VMC from current height "
                        f"{self.vmc_initial_height:.3f} m to "
                        f"{self.vmc_target_height:.3f} m, then "
                        f"{self.vmc_low_height:.3f} m and "
                        f"{self.vmc_high_height:.3f} m."
                    )

                t_vmc = t - self.vmc_start_time
                desired_height = self.get_scheduled_vmc_height(t_vmc)
                xyz_des = self.get_desired_foot_targets(desired_height)

                com, foot_pos = self.get_level_statics()

                roll, pitch, yaw = self.rpy_from_quat(self.imu_quat)
                roll_rate = self.imu_gyro[0]
                pitch_rate = self.imu_gyro[1]

                roll_err = self.roll_ref - roll
                pitch_err = self.pitch_ref - pitch

                tau_roll_des = self.K_roll_body * roll_err - self.D_roll_body * roll_rate
                tau_pitch_des = self.K_pitch_body * pitch_err - self.D_pitch_body * pitch_rate

                fz_dist = self.compute_fz_distribution(
                    com,
                    foot_pos,
                    tau_roll_des,
                    tau_pitch_des
                )

                leg_errors = []
                tau_cmd_all = np.zeros(12)
                blend_in = self.smoothstep(t_vmc / self.vmc_blend_time)
                blend_out = self.smoothstep((vmc_end - t) / self.vmc_exit_blend_time)
                torque_blend = min(blend_in, blend_out)
                joint_kp = (
                    (1.0 - blend_in) * 60.0
                    + blend_in * (
                        blend_out * self.vmc_joint_kp
                        + (1.0 - blend_out) * self.exit_joint_kp
                    )
                )
                joint_kd = (
                    (1.0 - blend_in) * 5.0
                    + blend_in * (
                        blend_out * self.vmc_joint_kd
                        + (1.0 - blend_out) * self.exit_joint_kd
                    )
                )

                R_body_to_world = self.quat_to_rotmat(self.imu_quat)

                for i in range(4):
                    idx = i * 3

                    q_leg = self.joint_q[idx:idx+3]
                    dq_leg = self.joint_dq[idx:idx+3]

                    J = self.jacobian_xyz(q_leg)

                    xyz = self.foot_relative_xyz(q_leg)
                    dxyz = J @ dq_leg

                    err_xyz = xyz_des[i] - xyz
                    leg_errors.append(np.linalg.norm(err_xyz))

                    F = np.zeros(3)

                    F[0] = self.Kx * err_xyz[0] - self.Dx * dxyz[0]
                    F[1] = self.Ky * err_xyz[1] - self.Dy * dxyz[1]
                    F[2] = self.Kz * err_xyz[2] - self.Dz * dxyz[2]

                    F_world_grav = np.array([0.0, 0.0, -fz_dist[i]])
                    F_body_grav = R_body_to_world.T @ F_world_grav
                    F += F_body_grav

                    tau_leg = J.T @ F
                    tau_cmd_all[idx:idx+3] = tau_leg

                tau_cmd_all = np.clip(tau_cmd_all, -self.tau_limits, self.tau_limits)

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = float(self.vmc_joint_ref[i])
                    self.cmd.motor_cmd[i].kp = float(joint_kp)
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = float(joint_kd)
                    self.cmd.motor_cmd[i].tau = float(
                        self.vmc_tau_scale * torque_blend * tau_cmd_all[i]
                    )

                if t - self.last_print_time >= 1.0:
                    self.last_print_time = t

                    mean_err = np.mean(leg_errors)
                    max_err = np.max(leg_errors)

                    print(
                        f"[{t:.1f}s] VMC hold | torque_blend={torque_blend:.2f}, "
                        f"h={desired_height:.3f} m | "
                        f"pitch={np.degrees(pitch):+.2f} deg, "
                        f"roll={np.degrees(roll):+.2f} deg | "
                        f"fz=({fz_dist[0]:.1f}, {fz_dist[1]:.1f}, {fz_dist[2]:.1f}, {fz_dist[3]:.1f}) N | "
                        f"err mean={mean_err*1000:.1f} mm, max={max_err*1000:.1f} mm | "
                        f"tau max={np.max(np.abs(tau_cmd_all)):.1f} Nm"
                    )

            # -------------------------
            # Phase 3: lower to crouch before damping.
            # -------------------------
            elif t < stand_down_end:
                phase = self.smoothstep((t - vmc_end) / self.stand_down_duration)
                q_des = (1.0 - phase) * self.vmc_joint_ref + phase * self.target_pose_1

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = float(q_des[i])
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 60.0
                    self.cmd.motor_cmd[i].kd = 5.0
                    self.cmd.motor_cmd[i].tau = 0.0

                if t - self.last_print_time >= 1.0:
                    self.last_print_time = t
                    print(f"[{t:.1f}s] lowering to targetPose 1 before damping")

            # -------------------------
            # Phase 4: damping mode.
            # -------------------------
            else:
                if not self.damping_started:
                    self.damping_started = True
                    print("VMC finished. Switching to damping mode.")

                self.send_damping_command(kd=2.0)

                if t - self.last_print_time >= 2.0:
                    self.last_print_time = t
                    print(f"[{t:.1f}s] damping mode")

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            
            elapsed = time.perf_counter() - (t + self.start_time)
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

if __name__ == "__main__":
    sim = "--sim" in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "--sim"]
    interface = args[0] if args else None

    if not sim:
        print("WARNING: REAL ROBOT LOW-LEVEL CONTROL.")
        input("Press Enter to continue...")

    controller = Go2Controller(interface=interface, sim=sim)

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nShutdown: damping then zero torque.")
        controller.send_damping_command(kd=2.0)
        controller.cmd.crc = controller.crc.Crc(controller.cmd)
        controller.pub.Write(controller.cmd)
        time.sleep(0.1)
        controller.send_zero_torque()
