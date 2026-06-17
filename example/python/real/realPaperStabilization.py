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
            self.mass =  1.*float(np.sum(self.mj_model.body_mass))
            self._extract_robot_geometry()
            self.time_up=5.
            print(
                f"Robot Info: Mass={self.mass:.2f}kg, "
                f"Offset Thigh={self.L_offset:.3f}m, Thigh={self.L_THIGH:.3f}m, Calf={self.L_CALF:.3f}m"
            )
            self.body_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                "world"
            )

            if self.body_id == -1:
                raise RuntimeError("world body not found")
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
            self.stand_down_joint_pos = np.array([0.04, 1.22, -2.44] * 4)
            self.stand_up_joint_pos_target = np.array([0.0, 0.67, -1.3] * 4)

        except Exception as e:
            print(f"Critical MuJoCo Load Error: {e}")
            sys.exit(1)
        ### Initialize state variables
        z_body = self.mj_data.xpos[self.body_id][0:3]
        print(f"Initial body height: {z_body}m")
        self.joint_q = np.zeros(12)
        self.joint_dq = np.zeros(12)

        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_gyro = np.zeros(3)

        self.dt = 0.002
        self.state_received = False
        self.prev_com_x = None
        self.prev_com_y = None
        self.prev_com_z = None
        self.prev_time = None
        ### Initialize CRC calculator
        self.crc = CRC()

        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.init_low_cmd()

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_handler, 10)
        self.flag=False

        ### Control targets
        
        self.roll_des=0.0
        self.pitch_des=0.0
        self.yaw_des=0.0
        self.x_des=0.0
        self.y_des=0.
        self.z_des=0.3

        ## 2 legs balance parameters
        self.stance_legs =["FR", "RL"] # Diagonal stance
        self.swing_legs = ["FL", "RR"]
        self.leg_index = {
            "FR": 0,
            "FL": 1,
            "RR": 2,
            "RL": 3,
        }
        self.swing_target_pos = {
            "FR": np.zeros(3),
            "FL": np.zeros(3),
            "RR": np.zeros(3),
            "RL": np.zeros(3),
        }
        self.lift_height = 0.2
        self.two_leg_mode_initialized = False

        print("sim =", sim)
        if not sim:
            self.release_motion_mode()
        else:
            print("Simulation mode: skipping MotionSwitcherClient/SportClient release.")

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
    def _extract_robot_geometry(self):
        thigh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_thigh")
        calf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        self.L_offset= np.linalg.norm(self.mj_model.body_pos[thigh_id])
        self.L_THIGH = np.linalg.norm(self.mj_model.body_pos[calf_id])
        foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot")
        self.L_CALF = np.linalg.norm(self.mj_model.body_pos[foot_id])
    def estimate_com_position(self):
        com, foot_positions = self.get_level_statics()
        mean_foot_x = np.mean([foot[0] for foot in foot_positions])
        com_x = com[0] - mean_foot_x
        mean_foot_y = np.mean([foot[1] for foot in foot_positions])
        com_y = com[1] - mean_foot_y
        mean_foot_z = np.mean([foot[2] for foot in foot_positions])
        com_z= com[2] - mean_foot_z

        return com_x, com_y, com_z
    def initialize_two_leg_mode(self, swing_legs, lift_height=0.06):
        for leg in swing_legs:
            leg_id = self.leg_index[leg]
            idx = 3 * leg_id

            q_leg = self.joint_q[idx:idx+3]
            p_now = self.fk_leg(q_leg, leg)

            p_target = p_now.copy()
            print(f"swing target before {leg}: {p_target}")
            p_target[2] += self.lift_height  # lift foot upward

            self.swing_target_pos[leg] = p_target
            print(f"swing target after{leg}: {p_target}")

        self.two_leg_mode_initialized = True
    def stance_forces_two_legs(self, Fz, stance_legs):
        forces = {}

        for leg in stance_legs:
            forces[leg] = np.array([0.0, 0.0, Fz / 2.0])

        return forces
    def swing_foot_vmc(self, leg, p_des):
        leg_id = self.leg_index[leg]
        idx = 3 * leg_id

        q_leg = self.joint_q[idx:idx+3]
        dq_leg = self.joint_dq[idx:idx+3]

        p = self.fk_leg(q_leg, leg)
        J = self.jacobian_leg(q_leg)

        v = J @ dq_leg

        Kp_cart = np.diag([100.0, 10.0, 300.0])
        Kd_cart = np.diag([3.0, 3.0, 10.0])

        F = Kp_cart @ (p_des - p) + Kd_cart @ (0.0 - v)

        tau = J.T @ F

        return tau
    def init_low_cmd(self):
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

    def low_state_handler(self, msg: LowState_):
        for i in range(12):
            self.joint_q[i] = msg.motor_state[i].q
            self.joint_dq[i] = msg.motor_state[i].dq

        self.imu_quat[:] = msg.imu_state.quaternion
        self.imu_gyro[:] = msg.imu_state.gyroscope

        self.state_received = True
    def get_level_statics(self):
        self.mj_data.qpos[0:3] = 0.0 #put the floating base at the world origin for statics computation
        self.mj_data.qpos[3:7] = self.imu_quat #use the IMU quaternion for frame orientation
        self.mj_data.qpos[self.qpos_adr] = self.joint_q #update the joint angles from the latest state

        mujoco.mj_forward(self.mj_model, self.mj_data) #use MuJoCo's kinematics to compute the positions of all bodies based on the current state

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
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.
            self.cmd.motor_cmd[i].kd = kd
            self.cmd.motor_cmd[i].tau = 0.0

    def send_zero_torque(self):
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
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
    def fk_leg(self, q, leg_name):
        q1, q2, q3 = q

        l1 = self.L_offset
        l2 = self.L_THIGH
        l3 = self.L_CALF

        # right legs: FR, RR -> -1
        # left legs: FL, RL -> +1
        side = -1.0 if leg_name in ["FR", "RR"] else 1.0

        x = l2*np.sin(q2) + l3*np.sin(q2 + q3)

        y = side*l1 \
            + l2*np.sin(q1)*np.cos(q2) \
            + l3*np.sin(q1)*np.cos(q2 + q3)

        z = -l2*np.cos(q1)*np.cos(q2) \
            - l3*np.cos(q1)*np.cos(q2 + q3)

        return np.array([x, y, z])
    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
    def jacobian_leg(self, q):
        q1, q2, q3 = q

        l2 = self.L_THIGH
        l3 = self.L_CALF

        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)

        J = np.array([
            [
                0.0,
                l2*c2 + l3*c23,
                l3*c23
            ],
            [
                l2*c1*c2 + l3*c1*c23,
                -l2*s1*s2 - l3*s1*s23,
                -l3*s1*s23
            ],
            [
                l2*s1*c2 + l3*s1*c23,
                l2*c1*s2 + l3*c1*s23,
                l3*c1*s23
            ]
        ])

        return J
    def attitude_errors(self):
        roll, pitch, yaw = self.quat_to_rpy(self.imu_quat)
        gyro = self.imu_gyro
        roll_error = self.roll_des - roll
        pitch_error = self.pitch_des - pitch
        yaw_error = self.yaw_des - yaw

        ed_roll  = 0.0 - gyro[0]
        ed_pitch = 0.0 - gyro[1]
        ed_yaw   = 0.0 - gyro[2]
        return roll_error, pitch_error, yaw_error, ed_roll, ed_pitch, ed_yaw
    def compute_VMC(self):
        com_x, com_y, com_z = self.estimate_com_position()
        roll_error, pitch_error, yaw_error, ed_roll, ed_pitch, ed_yaw = self.attitude_errors()
        kz = 2000.0
        dz = 10.0
        if self.prev_com_x is not None and self.prev_com_y is not None and self.prev_com_z is not None and self.prev_time is not None:
            dt = time.perf_counter() - self.prev_time
            if dt > 0:
                vx = (com_x - self.prev_com_x) / dt
                vy = (com_y - self.prev_com_y) / dt
                vz = (com_z - self.prev_com_z) / dt
        else:
            vx = 0.0
            vy = 0.0
            vz = 0.0

        self.prev_com_x = com_x
        self.prev_com_y = com_y
        self.prev_com_z = com_z
        self.prev_time = time.perf_counter()

        kx= 300.0
        dx = 10.0

        ky = 300.
        dy = 10.0

        k_roll = 100.0
        d_roll = 10.0

        k_pitch = 100.0
        d_pitch = 10.0

        # now ignore yaw
        k_yaw = 0.0
        d_yaw = 0.0
        e_x = self.x_des - com_x
        e_y = self.y_des - com_y
        e_z = self.z_des- com_z
        Fz = self.mass * 9.81 + kz * e_z  - dz *vz

        Fx = kx * e_x - dx *vx
        Fy = ky * e_y - dy *vy

        Tx = k_roll * roll_error     + d_roll * ed_roll
        Ty = k_pitch * pitch_error + d_pitch * ed_pitch
        Tz = k_yaw * yaw_error + d_yaw * ed_yaw

        return -Fx, Fy, Fz, Tx, Ty, Tz
    def distribute_vertical_forces(self, Fz, Tx, Ty, com, foot_positions):


        A = []

        for foot in foot_positions:
            r = foot - com
            x = r[0]
            y = r[1]

            A.append([1.0, y, -x])

        A = np.array(A).T  # shape 3x4

        b = np.array([Fz, Tx, Ty])


        Fzi = A.T @ np.linalg.inv(A @ A.T) @ b

        Fzi = np.clip(Fzi, 0.0, 120.0)

        return Fzi

    def quat_to_rpy(self,quat):
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z
    def paper_stance_control(self, stance_legs):
        roll, pitch, yaw = self.quat_to_rpy(self.imu_quat)
        gyro = self.imu_gyro

        roll_error = self.roll_des - roll
        pitch_error = self.pitch_des - pitch
        yaw_error = self.wrap_to_pi(self.yaw_des - yaw)

        roll_rate = gyro[0]
        pitch_rate = gyro[1]
        yaw_rate = gyro[2]

        # Body force
        Fx_body = 0.0
        Fy_body = 0.0
        Fz_body = self.mass * 9.81 

        # Split force equally between stance legs
        forces = {}
        for leg in stance_legs:
            forces[leg] = np.array([
                0.5 * Fx_body,
                0.5 * Fy_body,
                0.5 * Fz_body,
            ])

        # Roll: extra hip abduction torque
        roll_tau = (
            self.k_roll * roll_error
            + self.d_roll * (0.0 - roll_rate)
        )

        # Pitch: height target modification
        pitch_term = (
            self.k_pitch * np.sin(pitch_error)
            + self.d_pitch * (0.0 - pitch_rate)
        )

        for leg in stance_legs:
            if leg in ["FR", "FL"]:
                forces[leg][2] += -self.kz_pitch * pitch_term
            else:
                forces[leg][2] += +self.kz_pitch * pitch_term

        # Yaw: opposite forward forces
        yaw_torque = (
            self.k_yaw * yaw_error
            + self.d_yaw * (0.0 - yaw_rate)
        )

        yaw_force = yaw_torque / self.body_width

        leg_a, leg_b = stance_legs
        forces[leg_a][0] += yaw_force
        forces[leg_b][0] -= yaw_force

        # Convert to torques
        tau_all = np.zeros(12)

        for leg in stance_legs:
            leg_id = self.leg_index[leg]
            idx = 3 * leg_id

            q_leg = self.joint_q[idx:idx+3]
            J = self.jacobian_leg(q_leg)

            tau_leg = -J.T @ forces[leg]

            # extra roll control on hip abduction
            tau_leg[0] -= roll_tau

            tau_all[idx:idx+3] = tau_leg

        return tau_all
    
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
            elif t < self.time_up + 5.0: 
                if not self.flag:
                    print("Reached stand-up position, switching to VMC control.")
                    self.flag=True

                com_pos, foot_pos= self.get_level_statics()
                Fx, Fy, Fz, Tx, Ty,Tz=self.compute_VMC()
                Fzi = self.distribute_vertical_forces(Fz, Tx, Ty, com_pos, foot_pos)
                legs = ["FR", "FL", "RR", "RL"]
                tau_all = np.zeros(12)

                for leg_id, leg in enumerate(legs):
                    idx = 3 * leg_id

                    q_leg = self.joint_q[idx:idx+3]
                    J = self.jacobian_leg(q_leg)

                    F_foot = np.array([Fx/4, Fy/4, Fzi[leg_id]])

                    # Sign may need testing
                    tau_leg = -J.T @ F_foot

                    tau_all[idx:idx+3] = tau_leg
                
                tau_limit = 25.0
                tau_all = np.clip(tau_all, -tau_limit, tau_limit)

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01


                    self.cmd.motor_cmd[i].q = 0.
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 0.0
                    self.cmd.motor_cmd[i].kd = 1.2

                    # Add VMC torque
                    self.cmd.motor_cmd[i].tau = tau_all[i]
            else:


                if not self.two_leg_mode_initialized:
                    self.initialize_two_leg_mode(self.swing_legs, lift_height=0.04)
                    print("Initialized two-leg mode")

                # Body VMC
                tau_all = self.paper_stance_control(self.stance_legs)

                # Safety limit
                tau_limit = 25.0
                tau_all = np.clip(tau_all, -tau_limit, tau_limit)

                for i in range(12):
                    self.cmd.motor_cmd[i].mode = 0x01
                    self.cmd.motor_cmd[i].q = self.stand_up_joint_pos_target[i] 
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kp = 5.0
                    self.cmd.motor_cmd[i].kd = 1.0
                    self.cmd.motor_cmd[i].tau = tau_all[i]

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            com, foot_pos = self.get_level_statics()
            com_x, com_y, com_z = self.estimate_com_position()


            





            elapsed = time.perf_counter() - (t + self.start_time)
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

if __name__ == "__main__":
    sim = True
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

        controller.send_zero_torque()