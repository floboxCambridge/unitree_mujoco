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
            self.mass =  1.1* float(np.sum(self.mj_model.body_mass))

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
        self.pitch_ref = 0.0 * np.pi/180

        self.start_joint_pos = np.zeros(12)
        self.damping_started = False
        self.start_time = None
        self.vmc_start_time = None
        self.last_print_time = 0.0

        # ------------------------------------------------------------
        # Control timing.
        # ------------------------------------------------------------
        self.g = 9.81
        self.dt = 0.002

        self.time_up = 5.0
        self.stand_down_duration = 2.0
        self.damping_duration = 2.
        self.control_phase = "ramp"

        self.K_com_x = 200.0
        self.D_com_x = 15.0
        self.max_com_fx = 50.0
        self.prev_com_x = None

        # ------------------------------------------------------------
        # VMC PD gains: kept identical to stand_go2_grav_compensation.py.
        # ------------------------------------------------------------
        self.Kx, self.Dx = 200.0, 10.0
        self.Ky, self.Dy = 125.0, 12.0
        self.Kz, self.Dz = 300.0, 10.0

        self.K_roll_body = 25.0
        self.D_roll_body = 2.0

        self.K_pitch_body = 50.0
        self.D_pitch_body = 5.0

        self.tau_limits = np.array([23.0, 23.0, 40.0] * 4)

        # ------------------------------------------------------------
        # Joint references.
        # ------------------------------------------------------------
        self.stand_down_joint_pos = np.array([0.04, 1.22, -2.44] * 4)
        self.stand_up_joint_pos_target = np.array([0.0, 0.67, -1.3] * 4)
        self.stand_down_start_joint_pos = None

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
        # Desired body heights: kept identical to stand_go2_grav_compensation.py.
        # ------------------------------------------------------------
        self.height_sequence = [0.35]
        self.height_period = 60.0          # change height every 10 seconds
        self.height_transition_time = 3.0  # smooth transition duration
        self.vmc_cycle_duration = self.height_period * len(self.height_sequence)

        self.xyz_nominal_flat = None
        self.Flag=False

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
        self.last_tau_debug_time = 0.0
        self.tau_debug_period = 0.50  # print every 100 ms

        self.prev_tau_cmd_sent = np.zeros(12)
        self.vmc_torque_ramp_duration = 5.0
        self.tau_filtered = np.zeros(12)
        self.tau_filter_alpha = 0.12
        self.max_tau_rate = np.array([120.0, 180.0, 180.0] * 4)
        self.prev_tau_sent_control = np.zeros(12)
        self.last_vmc_debug_time = 0.0
        self.vmc_debug_period = 0.1  # seconds

        # ------------------------------------------------------------
        # Release high-level motion mode once, after DDS is initialized.
        # ------------------------------------------------------------
        print("sim =", sim)
        if not sim:
            self.release_motion_mode()
        else:
            print("Simulation mode: skipping MotionSwitcherClient/SportClient release.")
    def print_vmc_debug(
    self,
    t,
    desired_height,
    actual_height,
    xyz_des_all,
    xyz_all,
    dxyz_all,
    err_xyz_all,
    F_pd_all,
    F_grav_all,
    F_total_all,
    tau_raw,
    tau_sent,
    fz_dist,
    torque_ramp,
    roll,
    pitch,
):
        """
        Detailed VMC debug print.

        Shows:
        - desired vs actual foot position
        - foot velocity
        - position error
        - Cartesian forces
        - raw torque before filters/limits
        - sent torque after filters/limits
        """
        return

        if t - self.last_vmc_debug_time < self.vmc_debug_period:
            return

        self.last_vmc_debug_time = t

        leg_names = ["FR", "FL", "RR", "RL"]
        joint_names = ["hip", "thigh", "calf"]

        print("\n======================= VMC DEBUG =======================")
        print(
            f"t={t:.3f}s | "
            f"height_des={desired_height*100:.1f}cm | "
            f"height_act={actual_height*100:.1f}cm | "
            f"height_err={(desired_height-actual_height)*1000:+.1f}mm | "
            f"roll={np.degrees(roll):+.2f}deg | "
            f"pitch={np.degrees(pitch):+.2f}deg | "
            f"torque_ramp={torque_ramp:.3f}"
        )

        for leg in range(4):
            idx = leg * 3

            xyz_des = xyz_des_all[leg]
            xyz = xyz_all[leg]
            dxyz = dxyz_all[leg]
            err = err_xyz_all[leg]
            F_pd = F_pd_all[leg]
            F_grav = F_grav_all[leg]
            F_total = F_total_all[leg]

            print(f"\n{leg_names[leg]}:")
            print(
                f"  foot_err  xyz = "
                f"[{err[0]*1000:+.1f}, {err[1]*1000:+.1f}, {err[2]*1000:+.1f}] mm"
            )
            print(
                f"  foot_vel dxyz = "
                f"[{dxyz[0]:+.4f}, {dxyz[1]:+.4f}, {dxyz[2]:+.4f}] m/s"
            )

            print(
                f"  fz_dist = {fz_dist[leg]:+.2f} N"
            )

            print(
                f"  F_pd    = "
                f"[{F_pd[0]:+.2f}, {F_pd[1]:+.2f}, {F_pd[2]:+.2f}] N"
            )
            print(
                f"  F_grav  = "
                f"[{F_grav[0]:+.2f}, {F_grav[1]:+.2f}, {F_grav[2]:+.2f}] N"
            )
            print(
                f"  F_total = "
                f"[{F_total[0]:+.2f}, {F_total[1]:+.2f}, {F_total[2]:+.2f}] N"
            )

            print("  torques:")
            for j in range(3):
                k = idx + j
                print(
                    f"    {joint_names[j]:>6s}: "
                    f"raw={tau_raw[k]:+8.3f} Nm -> "
                    f"sent={tau_sent[k]:+8.3f} Nm | "
                    f"q={self.joint_q[k]:+7.3f} rad | "
                    f"dq={self.joint_dq[k]:+7.3f} rad/s"
                )

        print("=========================================================\n")
    def print_torque_debug(self, t, tau_raw, tau_sent):
        """
        Print torque debug information.

        tau_raw  = torque before clipping
        tau_sent = torque after clipping, i.e. what is actually sent to motors
        """

        if t - self.last_tau_debug_time < self.tau_debug_period:
            return

        self.last_tau_debug_time = t

        leg_names = ["FR", "FL", "RR", "RL"]
        joint_names = ["hip", "thigh", "calf"]

        tau_delta = tau_sent - self.prev_tau_cmd_sent
        self.prev_tau_cmd_sent = tau_sent.copy()

        saturated = np.abs(tau_raw) > self.tau_limits

        print("\n================ TORQUE DEBUG ================")
        print(f"t = {t:.3f} s")
        print("Format per joint:")
        print("  raw -> sent Nm | dq rad/s | delta_tau Nm | SAT")

        for leg in range(4):
            idx = leg * 3
            print(f"\n{leg_names[leg]}:")

            for j in range(3):
                k = idx + j
                sat_text = "SAT" if saturated[k] else ""

                print(
                    f"  {joint_names[j]:>6s}: "
                    f"{tau_raw[k]:+8.3f} -> {tau_sent[k]:+8.3f} Nm | "
                    f"dq={self.joint_dq[k]:+8.3f} rad/s | "
                    f"dTau={tau_delta[k]:+8.3f} Nm | "
                    f"{sat_text}"
                )

        print("==============================================\n")
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

        return h0 #(1-alpha) * h0 + alpha * h1
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
            perturb=0.
            if self.Flag == True and (i==0 or i==1):
                perturb = 0.1
            target[2] = -desired_height + perturb
            targets.append(target)

        return targets
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
        self.mj_data.qpos[3:7] = self.imu_quat
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
    def distribute_fx_weighted(self, Fx_total, fz_dist):
        """
        Distribute horizontal force proportional to vertical load.
        This is safer for contact because loaded legs receive more horizontal force.
        """
        weights = fz_dist / max(1e-6, np.sum(fz_dist))
        return Fx_total * weights
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
            should_stop = False
            
            if t < self.time_up:
                phase = np.tanh(t / 1.5)
                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = phase * self.stand_up_joint_pos_target[i] + (1-phase) * self.stand_down_joint_pos[i]
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp, self.cmd.motor_cmd[i].kd = (phase*60 + 20), 3.5
                    self.cmd.motor_cmd[i].tau = 0.0
            else:
                if self.xyz_nominal_flat is None:
                    self.xyz_nominal_flat = [
                        self.foot_relative_xyz(self.joint_q[i*3:i*3+3])
                        for i in range(4)
                    ]

                    self.vmc_start_time = t

                    print("3D VMC + Gravity Compensation Active.")
                    print("Height sequence active: 20 cm -> 30 cm -> 35 cm, changing every 10 s.")
                t_vmc = t - self.vmc_start_time

                if t_vmc < self.vmc_cycle_duration:
                    desired_height = self.get_desired_height(t_vmc)
                    xyz_des = self.get_desired_foot_targets(desired_height)
                    com, foot_pos = self.get_level_statics()
                    roll, pitch, yaw = self.rpy_from_quat(self.imu_quat)
                    roll_rate = self.imu_gyro[0]
                    pitch_rate = self.imu_gyro[1]

                    roll_err = self.roll_ref - roll
                    pitch_err = self.pitch_ref - pitch

                    tau_roll_des = self.K_roll_body * roll_err - self.D_roll_body * roll_rate
                    tau_pitch_des = self.K_pitch_body * pitch_err - self.D_pitch_body * pitch_rate
                    # First compute nominal vertical support without horizontal correction.
                    fz_dist = self.compute_fz_distribution(
                        com,
                        foot_pos,
                        tau_roll_des,
                        tau_pitch_des,
                    )

                    # COM x spring-damper.
                    foot_center_x = np.mean([p[0] for p in foot_pos])
                    com_x = com[0]
                    x_err_com = foot_center_x - com_x

                    if self.prev_com_x is None:
                        com_vx = 0.0
                    else:
                        dt_com = max(1e-4, t - self.prev_com_time)
                        com_vx = (com_x - self.prev_com_x) / dt_com

                    self.prev_com_x = com_x
                    self.prev_com_time = t

                    Fx_total_com = self.K_com_x * x_err_com - self.D_com_x * com_vx
                    Fx_total_com = np.clip(Fx_total_com, -self.max_com_fx, self.max_com_fx)

                    # Distribute horizontal force over loaded feet.
                    Fx_leg = self.distribute_fx_weighted(Fx_total_com, fz_dist)

                    # Estimate pitch moment created by horizontal forces.
                    # r x F, pitch/y component = z*Fx - x*Fz.
                    # Here only the horizontal part:
                    tau_pitch_from_fx = 0.0
                    for i in range(4):
                        lever = foot_pos[i] - com
                        tau_pitch_from_fx += lever[2] * Fx_leg[i]

                    # Recompute vertical load distribution to cancel this pitch contribution.
                    fz_dist = self.compute_fz_distribution(
                        com,
                        foot_pos,
                        tau_roll_des,
                        tau_pitch_des - tau_pitch_from_fx,
                    )

                    # Recompute Fx distribution with updated fz.
                    Fx_leg = self.distribute_fx_weighted(Fx_total_com, fz_dist)
                    leg_errors = []
                    tau_cmd_all = np.zeros(12)
                    R_body_to_world = self.quat_to_rotmat(self.imu_quat)

                    vmc_elapsed = t - self.vmc_start_time
                    torque_ramp = self.smoothstep(vmc_elapsed / self.vmc_torque_ramp_duration)

                    xyz_all = []
                    dxyz_all = []
                    err_xyz_all = []
                    F_pd_all = []
                    F_grav_all = []
                    F_total_all = []

                    for i in range(4):
                        idx = i * 3

                        q_leg = self.joint_q[idx:idx+3]
                        dq_leg = self.joint_dq[idx:idx+3]

                        J = self.jacobian_xyz(q_leg)

                        xyz = self.foot_relative_xyz(q_leg)
                        dxyz = J @ dq_leg

                        err_xyz = xyz_des[i] - xyz
                        leg_errors.append(np.linalg.norm(err_xyz))

                        # Foot-position corrective force.
                        F_pd = np.zeros(3)
                        F_pd[0] = self.Kx * err_xyz[0] - self.Dx * dxyz[0]
                        F_pd[1] = self.Ky * err_xyz[1] - self.Dy * dxyz[1]
                        F_pd[2] = self.Kz * err_xyz[2] - self.Dz * dxyz[2]
                        f_perturb=0.
                        if t>20.0 and t<25.0:
                            f_perturb = 0.0
                            self.Flag=False
                        else:
                            self.Flag=False
                        F_world_grav = np.array([0.0, 0.0, -fz_dist[i] + f_perturb])

                        # Sign may need flipping after testing.
                        F_world_com = np.array([Fx_leg[i], 0.0, 0.0])

                        F_world_total = F_world_grav + F_world_com
                        F_body_total = R_body_to_world.T @ F_world_total

                        F = F_body_total
                        F_body_grav = R_body_to_world.T @ F_world_grav

                        # Ramp only the corrective part, not the support part.
                        F = F_body_grav  + torque_ramp * F_pd
                        xyz_all.append(xyz.copy())
                        dxyz_all.append(dxyz.copy())
                        err_xyz_all.append(err_xyz.copy())
                        F_pd_all.append(F_pd.copy())
                        F_grav_all.append(F_body_grav.copy())
                        F_total_all.append(F.copy())

                        tau_leg = J.T @ F
                        tau_cmd_all[idx:idx+3] = tau_leg
                    actual_height = -np.mean([xyz_all[i][2] for i in range(4)])
                    height_error = desired_height - actual_height
                    if t - self.last_print_time >= self.vmc_debug_period:
                        self.last_print_time = t

                        mean_err = np.mean(leg_errors)
                        max_err = np.max(leg_errors)

                        print(
                        f"[{t:.1f}s] "
                        f"height_des={xyz_des} cm, "
                        f"height_act={actual_height*100:.1f} cm, "
                        f"h_err={height_error*1000:+.1f} mm, "
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
                        f"torque_ramp={torque_ramp:.3f}"
                        f"com_x={com_x:+.3f}, foot_cx={foot_center_x:+.3f}, "
                        f"x_err_com={x_err_com*1000:+.1f} mm, "
                        f"Fx_total={Fx_total_com:+.1f} N, "
                        f"tau_pitch_fx={tau_pitch_from_fx:+.2f} Nm | "
                    )
                    
                    tau_raw = tau_cmd_all.copy()
                    tau_clipped = np.clip(tau_raw, -self.tau_limits, self.tau_limits)

                    """
                    self.tau_filtered = (
                        (1.0 - self.tau_filter_alpha) * self.tau_filtered
                        + self.tau_filter_alpha * tau_clipped
                    )

                    tau_sent = self.tau_filtered.copy()
                    max_delta = self.max_tau_rate * self.dt

                    tau_sent = np.clip(
                        tau_sent,
                        self.prev_tau_sent_control - max_delta,
                        self.prev_tau_sent_control + max_delta,
                    )

                    self.prev_tau_sent_control = tau_sent.copy()
                    """
                    tau_sent = tau_clipped.copy()

                    self.print_vmc_debug(
                            t=t,
                            desired_height=desired_height,
                            actual_height=actual_height,
                            xyz_des_all=xyz_des,
                            xyz_all=xyz_all,
                            dxyz_all=dxyz_all,
                            err_xyz_all=err_xyz_all,
                            F_pd_all=F_pd_all,
                            F_grav_all=F_grav_all,
                            F_total_all=F_total_all,
                            tau_raw=tau_raw,
                            tau_sent=tau_sent,
                            fz_dist=fz_dist,
                            torque_ramp=torque_ramp,
                            roll=roll,
                            pitch=pitch,
                        )
                    
                    for i in range(12):
                        self.cmd.motor_cmd[i].q, self.cmd.motor_cmd[i].kp = 0.0, 0.0
                        self.cmd.motor_cmd[i].dq, self.cmd.motor_cmd[i].kd = 0.0, 1.2
                        self.cmd.motor_cmd[i].tau = float(tau_sent[i])

                elif t_vmc < self.vmc_cycle_duration + self.stand_down_duration:
                    if self.stand_down_start_joint_pos is None:
                        self.stand_down_start_joint_pos = self.joint_q.copy()
                        print("Full VMC cycle finished. Crouching down before damping.")

                    phase = self.smoothstep((t_vmc - self.vmc_cycle_duration) / self.stand_down_duration)
                    q_des = (1.0 - phase) * self.stand_down_start_joint_pos + phase * self.stand_down_joint_pos

                    for i in range(12):
                        self.cmd.motor_cmd[i].mode = 0x01
                        self.cmd.motor_cmd[i].q = float(q_des[i])
                        self.cmd.motor_cmd[i].dq = 0.0
                        self.cmd.motor_cmd[i].kp = 60.0
                        self.cmd.motor_cmd[i].kd = 3.5
                        self.cmd.motor_cmd[i].tau = 0.0

                else:
                    if not self.damping_started:
                        self.damping_started = True
                        print("Crouch complete. Switching to damping mode.")

                    #self.send_damping_command(kd=2.0)
                    should_stop = t_vmc >= (
                        self.vmc_cycle_duration
                        + self.stand_down_duration
                        + self.damping_duration
                    )

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            
            elapsed = time.perf_counter() - (t + self.start_time)
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

            if should_stop:
                print("Simulation cycle complete.")
                break

if __name__ == "__main__":
    sim = False
    args = [arg for arg in sys.argv[1:] if arg != "--sim"]
    interface = args[0] if args else None

    if not sim:
        print("WARNING: REAL ROBOT LOW-LEVEL CONTROL.")
        input("Press Enter to continue...")

    controller = Go2Controller(interface=interface, sim=sim)
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nShutdown.")
        controller.send_damping_command(kd=2.0)
        controller.cmd.crc = controller.crc.Crc(controller.cmd)
        controller.pub.Write(controller.cmd)
        time.sleep(0.1)
        controller.send_zero_torque()