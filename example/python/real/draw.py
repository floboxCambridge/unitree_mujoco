import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulate_python"))

from _unitree_sdk_path import ensure_unitree_sdk2py

ensure_unitree_sdk2py()

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class Go2FRForwardStepController:
    def __init__(self, interface=None, sim=True):
        resolved_interface = interface if interface is not None else os.getenv("UNITREE_INTERFACE", "lo")
        if interface is None:
            ChannelFactoryInitialize(1, resolved_interface)
        else:
            ChannelFactoryInitialize(0, interface)

        self.model_path = Path(__file__).resolve().parents[3] / "unitree_robots/go2/go2.xml"
        self.unitree_joint_names = [ #from mujoco model names
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        self.leg_names = ["FR", "FL", "RR", "RL"] #for control purposes
        self.leg_index = {name: idx for idx, name in enumerate(self.leg_names)}
        self.swing_sequence = ["FR"]

        self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mass = float(np.sum(self.mj_model.body_mass))
        self._extract_robot_geometry()

        self.joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.unitree_joint_names
        ]
        self.qpos_adr = np.array([self.mj_model.jnt_qposadr[jid] for jid in self.joint_ids])
        self.foot_body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.foot_body_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.foot_body_names
        ]

        self.dt = 0.002
        self.time_up = 5.0 #time for the robot to rise
        self.stand_down_joint_pos = np.array([0.04, 1.22, -2.44] * 4) #positions extracted from the provided default code for standing robot
        self.stand_up_joint_pos_target = np.array([0.0, 0.67, -1.3] * 4)

        self.roll_des = 0.0 #desired positions at equilibrium
        self.pitch_des = 0.0
        self.yaw_des = 0.0
        self.x_des = 0.0
        self.y_des = 0.0
        self.z_des = 0.30 #desired standing height

        self.nominal_x_des = 0.0 #position of the com after the walk
        self.nominal_y_des = 0.0


        self.initial_stabilize_duration = 2.5
        self.z_compression = 0.15
        self.z_extension = 0.4
        self.z_rest=0.3

        self.body_tau_limit = 30.0
        self.swing_tau_limit = 30.0

        self.prev_com = None
        self.prev_time = None
        self.state_received = False
        self.phase_announced = None
        self.fr_bend_home = None

        # Virtual spring parameters for bending toward FR
        self.fr_bend_kp = np.diag([80.0, 80.0, 1000.0])
        self.fr_bend_kd = np.diag([6.0, 6.0, 10.0])


        self.fr_phase = "compress"

        self.fr_compression_distance = 0.2 # 15 cm
        self.fr_extension_distance = 0.4   # 35 cm
        self.fr_done_distance = 0.33

        # Reference while compressing FR.
        # Since z is usually negative downward in your FK, positive z offset brings foot closer to hip.
        self.fr_compress_offset = np.array([0.0, 0.0, -0.20])

        # Reference while jumping/extending FR.
        # Negative z offset sends the foot farther away from the hip.
        self.fr_jump_offset = np.array([0.0, 0.0, +0.20])

        # Start with compression reference
        self.fr_bend_offset = self.fr_compress_offset.copy()
        self.fr_bend_tau_limit = 30.0
        self.fr_bend_force_limit = 30.0

        self.swing_home = {}
        self.swing_lifted = {}
        self.swing_forward = {}
        self.swing_compressed = {}
        self.Kx = 200.
        self.Dx =5.
        self.Ky = 350.
        self.Dy = 7.
        self.Kz = 1000.
        self.Dz = 20.
        self.K_roll = 150.
        self.K_pitch = 150.
        self.K_yaw = 35.
        self.D_roll = 10.
        self.D_pitch = 10.
        self.D_yaw = 3.
        self.horizontal_force_limit = 60.0

        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_gyro = np.zeros(3)

        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.init_low_cmd()

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)

        if not sim:
            self.release_motion_mode()
        else:
            print("Simulation mode: skipping MotionSwitcherClient/SportClient release.")

    def _extract_robot_geometry(self): #extract measurement for FK from mujoco model
        thigh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_thigh")
        calf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot")
        self.L_offset = np.linalg.norm(self.mj_model.body_pos[thigh_id])
        self.L_THIGH = np.linalg.norm(self.mj_model.body_pos[calf_id])
        self.L_CALF = np.linalg.norm(self.mj_model.body_pos[foot_id])

    def init_low_cmd(self): #init the state receiver 
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

    def low_state_handler(self, msg: LowState_): #takes care of the state receiver
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq
        self.imu_quat[:] = msg.imu_state.quaternion
        self.imu_gyro[:] = msg.imu_state.gyroscope
        self.state_received = True   

    def release_motion_mode(self): #release the control via the remote for interfacing with the robot
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

    def get_level_statics(self): #receive state informations and store them
        self.mj_data.qpos[0:3] = 0.0
        self.mj_data.qpos[3:7] = self.imu_quat
        self.mj_data.qpos[self.qpos_adr] = self.joint_q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        com = self.compute_whole_body_com()
        foot_pos = [self.mj_data.xpos[bid].copy() for bid in self.foot_body_ids]
        return com, foot_pos

    def compute_whole_body_com(self): #compute the position of the center of mass in a relative frame 
        masses = self.mj_model.body_mass
        xpos = self.mj_data.xipos
        total_mass = np.sum(masses[1:])
        return np.sum(xpos[1:] * masses[1:, None], axis=0) / total_mass

    def estimate_com_position(self):
        com, foot_positions = self.get_level_statics()
        foot_mean = np.mean(np.array(foot_positions), axis=0)
        return com - foot_mean

    def fk_leg(self, q, leg_name): #forward kinematics controller
        q1, q2, q3 = q
        side = -1.0 if leg_name in ["FR", "RR"] else 1.0
        x = self.L_THIGH * np.sin(q2) + self.L_CALF * np.sin(q2 + q3)
        y = side * self.L_offset
        y += self.L_THIGH * np.sin(q1) * np.cos(q2)
        y += self.L_CALF * np.sin(q1) * np.cos(q2 + q3)
        z = -self.L_THIGH * np.cos(q1) * np.cos(q2)
        z -= self.L_CALF * np.cos(q1) * np.cos(q2 + q3)
        return np.array([x, y, z])

    def jacobian_leg(self, q): #obtained by analytical derivation of the fk model 
        q1, q2, q3 = q
        s1, c1 = np.sin(q1), np.cos(q1)
        s2, c2 = np.sin(q2), np.cos(q2)
        s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
        return np.array([
            [0.0, self.L_THIGH * c2 + self.L_CALF * c23, self.L_CALF * c23],
            [
                self.L_THIGH * c1 * c2 + self.L_CALF * c1 * c23,
                -self.L_THIGH * s1 * s2 - self.L_CALF * s1 * s23,
                -self.L_CALF * s1 * s23,
            ],
            [
                self.L_THIGH * s1 * c2 + self.L_CALF * s1 * c23,
                self.L_THIGH * c1 * s2 + self.L_CALF * c1 * s23,
                self.L_CALF * c1 * s23,
            ],
        ])

    def quat_to_rpy(self, quat): #maps the quaternion measurements into the rotation along euler frame 
        w, x, y, z = quat
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(t2, -1.0, 1.0))
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def angle_error(self, target, measured):
        return np.arctan2(np.sin(target - measured), np.cos(target - measured))

    def compute_body_wrench(self):
        com = self.estimate_com_position()
        now = time.perf_counter()
        if self.prev_com is None or self.prev_time is None:
            com_vel = np.zeros(3)
        else:
            dt = max(now - self.prev_time, 1e-3)
            com_vel = (com - self.prev_com) / dt
        self.prev_com = com.copy()
        self.prev_time = now

        roll, pitch, yaw = self.quat_to_rpy(self.imu_quat)
        roll_error = self.roll_des - roll
        pitch_error = self.pitch_des - pitch
        yaw_error = self.angle_error(self.yaw_des, yaw)

        fx = self.Kx * (self.x_des - com[0]) - self.Dx * com_vel[0] # PD controller for position of the COM
        fy = self.Ky * (self.y_des - com[1]) - self.Dy * com_vel[1]
        fz = self.mass * 9.81 + self.Kz * (self.z_des - com[2]) - self.Dz * com_vel[2] #just try with smaller gain for kz
        tx = self.K_roll * roll_error - self.D_roll * self.imu_gyro[0]
        ty = self.K_pitch * pitch_error - self.D_pitch * self.imu_gyro[1]
        tz = self.K_yaw * yaw_error - self.D_yaw * self.imu_gyro[2]
        return -fx, fy, fz, tx, ty, tz

    def distribute_vertical_forces(self, total_fz, tx, ty, com, foot_positions, stance_legs): #solve lqr problem to find how to distribute vertical force  and the torques not around the vertical axis on the stance legs
        stance_ids = [self.leg_index[leg] for leg in stance_legs]
        A = []
        for idx in stance_ids:
            r = foot_positions[idx] - com
            A.append([1.0, r[1], -r[0]])
        A = np.array(A).T
        b = np.array([total_fz, tx, ty])
        raw = np.linalg.pinv(A) @ b
        raw = np.clip(raw, 0.0, 160.0)
        vertical = {leg: 0.0 for leg in self.leg_names}
        for leg, force in zip(stance_legs, raw):
            vertical[leg] = force
        return vertical

    def distribute_horizontal_forces(self, total_fx, total_fy, tz, com, foot_positions, stance_legs):
        stance_ids = [self.leg_index[leg] for leg in stance_legs]
        A = np.zeros((3, 2 * len(stance_ids)))
        for col, idx in enumerate(stance_ids):
            r = foot_positions[idx] - com
            A[0, 2 * col] = 1.0
            A[1, 2 * col + 1] = 1.0
            A[2, 2 * col] = -r[1]
            A[2, 2 * col + 1] = r[0]
        b = np.array([total_fx, total_fy, tz])
        raw = np.linalg.pinv(A) @ b
        raw = np.clip(raw, -self.horizontal_force_limit, self.horizontal_force_limit)
        horizontal = {leg: np.zeros(2) for leg in self.leg_names}
        for col, leg in enumerate(stance_legs):
            horizontal[leg] = raw[2 * col:2 * col + 2]
        return horizontal

    def compute_stance_torques(self, stance_legs): #computes torque for the stance legs
        com, foot_positions = self.get_level_statics()
        fx, fy, fz, tx, ty, tz = self.compute_body_wrench()
        vertical = self.distribute_vertical_forces(fz, tx, ty, com, foot_positions, stance_legs)
        horizontal = self.distribute_horizontal_forces(fx, fy, tz, com, foot_positions, stance_legs)
        tau_all = np.zeros(12)

        for leg in stance_legs:
            idx = 3 * self.leg_index[leg]
            q_leg = self.joint_q[idx:idx + 3]
            J = self.jacobian_leg(q_leg) #vmc 
            foot_force = np.array([horizontal[leg][0], horizontal[leg][1], vertical[leg]]) #we need to apply "special" force on z axis to avoid unbalance
            tau_all[idx:idx + 3] = -J.T @ foot_force

        return np.clip(tau_all, -self.body_tau_limit, self.body_tau_limit)

    def add_fr_bend_spring_torque(self, tau_all, phase):
        leg = "FR"
        idx = 3 * self.leg_index[leg]

        q_leg = self.joint_q[idx:idx + 3]
        dq_leg = self.joint_dq[idx:idx + 3]

        p_leg = self.fk_leg(q_leg, leg)
        J = self.jacobian_leg(q_leg)
        v_leg = J @ dq_leg

        # Save the FR foot position at the moment bending starts
        if self.fr_bend_home is None:
            self.fr_bend_home = p_leg.copy()

        # Desired FR foot position.
        # Positive x  -> forward
        # Negative y  -> right side, because FR/RR have side = -1 in fk_leg()
        # Positive z  -> leg bends/compresses upward in your FK convention
        if phase == "compress":
            fr_bend_offset = -0.2
        elif phase == "jump":
            fr_bend_offset = 0.3
        elif phase == "done":
            fr_bend_offset = -0.1

        target_pos = self.fr_bend_home + [0., 0., fr_bend_offset]

        # Virtual spring-damper force
        if phase == "compress":
            self.fr_bend_kp = np.diag([-100.0, -100.0, 300.0])
            self.fr_bend_kd = np.diag([6.0, 6.0, 30.0])
        elif phase == "jump":
            self.fr_bend_kp = np.diag([-80.0, -80.0, 2000.0])
            self.fr_bend_kd = np.diag([6.0, 6.0, 50.0])
        elif phase == "done":
            self.fr_bend_kp = np.diag([-100.0, -100.0, 200.0])
            self.fr_bend_kd = np.diag([6.0, 6.0, 5.0])
        spring_force = self.fr_bend_kp @ (target_pos - p_leg) - self.fr_bend_kd @ v_leg
        # Limit the extra force
        if phase == "jump" or phase == "done":
            spring_force = np.clip(
                spring_force,
                -120.,
                120.
            )
        else:
            spring_force = np.clip(
                spring_force,
                -80,
                80
            )


        # Convert foot force into joint torques.
        # Use the same sign convention as your stance VMC controller.
        tau_fr = -J.T @ spring_force

        # Optional extra safety limit only for this added spring torque
        tau_fr = np.clip(tau_fr, -self.fr_bend_tau_limit, self.fr_bend_tau_limit)

        tau_all[idx:idx + 3] += tau_fr

        return np.clip(tau_all, -self.body_tau_limit, self.body_tau_limit)

    
    def interpolate(self, start, end, alpha): #try to remove (or decrease interpolation period)
        alpha = np.clip(alpha, 0.0, 1.0)
        return end

    def support_legs_for(self, swing_leg):
        return [leg for leg in self.leg_names if leg != swing_leg]

    def shift_y_for(self, swing_leg):
        # Positive y shifts away from right legs, negative y shifts away from left legs.
        return abs(self.shift_y_des) if swing_leg in ["FR", "RR"] else -abs(self.shift_y_des)

    def get_swing_phase_target(self, swing_leg, phase_name):
        # defines target positions for VMC only when one leg is in the air

        if swing_leg not in self.swing_home:
            idx = 3 * self.leg_index[swing_leg]

            home_pos = self.fk_leg(
                self.joint_q[idx:idx + 3],
                swing_leg
            )

            self.swing_home[swing_leg] = home_pos

            # target position to lift/compress the leg
            self.swing_compressed[swing_leg] = home_pos + np.array(
                [0.0, 0.0, self.z_compression]
            )

        if phase_name == "compressed":
            return self.swing_compressed[swing_leg]

        return None

    def add_swing_torque(self, tau_all, swing_leg, phase_name):
        swing_target = self.get_swing_phase_target(swing_leg, phase_name)
        idx = 3 * self.leg_index[swing_leg]
        tau_all[idx:idx + 3] = self.jumping_foot_torque(swing_leg, swing_target)
        return tau_all

    def announce_phase(self, name): #for printing the current phase
        if self.phase_announced != name:
            print(f"phase -> {name}")
            self.phase_announced = name

    def command_stand_pose(self, phase): #for the initiation  (when the robot slowly rises)
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = phase * self.stand_up_joint_pos_target[i] + (1.0 - phase) * self.stand_down_joint_pos[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = phase * 60.0 + 20.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

    def apply_tau_command(self, tau_all, posture_kp=4.0, posture_kd=1.2):
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = self.stand_up_joint_pos_target[i] #for init
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kp = posture_kp
            self.cmd.motor_cmd[i].kd = posture_kd
            self.cmd.motor_cmd[i].tau = tau_all[i] #for vmc
    def send_damping_command(self, kd=2.0): #return to damping safe after end of experiment/emergency stop
        """
        Damping-only mode.

        This does not command a joint position.
        It only damps joint velocity.

        On the real robot this is much safer than abruptly sending zero torque,
        because the legs do not go completely passive instantly.
        """

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.
            self.cmd.motor_cmd[i].kd = kd
            self.cmd.motor_cmd[i].tau = 0.0
    def send_zero_torque(self): #to end the execution safely
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
    def run(self):
        print("Waiting for state...")
        while not self.state_received: #begin only if contact established with the robot
            time.sleep(0.01)

        start_time = time.perf_counter()
        sequence_start = start_time + self.time_up
        t_print = start_time
        while True:
            t = time.perf_counter()
            elapsed = t - start_time
            
            if elapsed < self.time_up: #first phase:ruse if the robot
                self.announce_phase("stand_up")
                phase = np.tanh(elapsed / 1.5)
                self.command_stand_pose(phase)
            else:
                seq_t = t - sequence_start
                tau_all = np.zeros(12)
                fr_idx = 3 * self.leg_index["FR"]
                fr_pose = self.fk_leg(self.joint_q[fr_idx:fr_idx + 3], "FR")
                fr_distance = fr_pose[2] #z position of the foot
                if t - t_print> 0.1: #print only once at the beginning of the sequence
                    t_print=t


                if seq_t < self.initial_stabilize_duration: #four legs just switch to vmc controller (position control to torque control)
                    self.announce_phase("four_leg_stabilize")
                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des
                    tau_all = self.compute_stance_torques(self.leg_names)
                else:
                    fr_idx = 3 * self.leg_index["FR"]
                    fr_pose = self.fk_leg(self.joint_q[fr_idx:fr_idx + 3], "FR")
                    fr_distance = np.linalg.norm(fr_pose)

                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des

                    # Keep all four legs in contact
                    tau_all = self.compute_stance_torques(self.leg_names)

                    if self.fr_phase == "compress":
                        self.announce_phase("fr_compress_spring")

                        self.K_pitch = 0.0
                        self.D_pitch = 10.0
                        self.K_roll = 0.0
                        self.D_roll = 6.0
                        self.K_yaw = 0.0
                        self.D_yaw = 0.0

                        # Use compression reference
                        self.fr_bend_offset = self.fr_compress_offset.copy()

                        # Add compression spring on FR
                        tau_all = self.add_fr_bend_spring_torque(tau_all, "compress")

                        # New case: when right compression is reached
                        if fr_distance <= self.fr_compression_distance:
                            print(f"FR compression reached: {fr_distance:.3f} m")
                            self.fr_phase = "jump"

                    elif self.fr_phase == "jump":
                        self.announce_phase("fr_jump_spring")

                        self.K_pitch = 0.0
                        self.D_pitch = 0.0
                        self.K_roll = 100.0
                        self.D_roll = 10.0
                        self.K_yaw = 0.0
                        self.D_yaw = 0.0

                        # Set reference much farther so FR extends hard
                        self.fr_bend_offset = self.fr_jump_offset.copy()

                        # Add jump spring on FR
                        tau_all = self.add_fr_bend_spring_torque(tau_all, "jump")

                        # Remove contractile element once FR reaches 35 cm
                        if fr_distance >= self.fr_extension_distance:
                            print(f"FR extension reached: {fr_distance:.3f} m")
                            self.fr_phase = "done"
                        
                    elif self.fr_phase == "done":
                        self.K_pitch = 0.0
                        self.D_pitch = 0.0
                        self.K_roll = 100.0
                        self.D_roll = 10.0
                        self.K_yaw = 0.0
                        self.D_yaw = 0.0
                        self.K_pitch = 0.0
                        self.D_pitch = 0.0
                        self.K_roll = 0.0
                        self.D_roll = 0.0
                        self.K_yaw = 0.0
                        self.D_yaw = 0.0

                        # Set reference much farther so FR extends hard
                        self.fr_bend_offset = self.fr_jump_offset.copy()

                        # Add jump spring on FR
                        tau_all = self.add_fr_bend_spring_torque(tau_all, "done")


                        if fr_distance <= self.fr_done_distance +0.01:
                            print(f"FR extension reached: {fr_distance:.3f} m")
                            self.fr_phase = "over"

                    else:
                        self.announce_phase("four_leg_restabilize")

                        # Contractile element removed here.
                        # Restore attitude stabilization.
                        self.K_pitch = 150.0
                        self.D_pitch = 10.0
                        self.K_roll = 150.0
                        self.D_roll = 10.0
                        self.K_yaw = 35.0
                        self.D_yaw = 3.0

                        self.x_des = self.nominal_x_des
                        self.y_des = self.nominal_y_des

                        # Only normal 4-leg stabilization, no FR spring
                        tau_all = self.compute_stance_torques(self.leg_names)

                """
                else:
                    self.announce_phase("four_leg_restabilize")
                    self.Kx =350.
                    self.Dx = 10.
                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des
                    tau_all = self.compute_stance_torques(self.leg_names)
                """
                self.apply_tau_command(tau_all)

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)

            loop_elapsed = time.perf_counter() - t
            if loop_elapsed < self.dt:
                time.sleep(self.dt - loop_elapsed)


if __name__ == "__main__":
    sim = "--sim" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--sim"]
    interface = args[0] if args else None

    if not sim:
        print("WARNING: REAL ROBOT LOW-LEVEL CONTROL.")
        input("Press Enter to continue...")

    controller = Go2FRForwardStepController(interface=interface, sim=sim)
    try:
        controller.run()
    except KeyboardInterrupt: #in case of emergency stop
        print("\nShutdown.")

        start_time_shutdown = time.perf_counter()

        while time.perf_counter() - start_time_shutdown < 3.0:
            t_shutdown = time.perf_counter() - start_time_shutdown

            # Start damped, then reduce smoothly
            kd_start = 2.0
            kd_end = 0.3
            alpha = min(t_shutdown / 3.0, 1.0)

            kd = (1.0 - alpha) * kd_start + alpha * kd_end

            controller.send_damping_command(kd=kd)
            controller.cmd.crc = controller.crc.Crc(controller.cmd)
            controller.pub.Write(controller.cmd)

            time.sleep(controller.dt)

        controller.send_zero_torque() #ends program