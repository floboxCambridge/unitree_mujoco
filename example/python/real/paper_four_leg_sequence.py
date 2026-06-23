import time

import numpy as np
import sys
from paper_fr_forward_step import Go2FRForwardStepController


class Go2FourLegSequenceController(Go2FRForwardStepController):
    def __init__(self, interface=None, sim=True):
        super().__init__(interface=interface, sim=sim)
        self.step_sequence = ["FR", "FL", "RR", "RL"]
        self.swing_lift_height = 0.07
        self.initial_stabilize_duration = 2.5
        self.inter_leg_stabilize_duration = 1.5
        self.final_stabilize_duration = 2.0
        self.sequence_done_hold = 3.0

        self.swing_home = None
        self.swing_lifted = None
        self.swing_forward = None
        self.active_leg_slot = None
        self.active_shift_target = None
        self.restabilize_hold_ratio = 0.35
        self.last_leg_com_shift_duration = 4.0
        self.last_leg_restabilize_hold_ratio = 0.6
        self.last_leg_step_scale = 0.65
        self.step_delta_x = -0.30

    def reset_swing_targets(self):
        self.swing_home = None
        self.swing_lifted = None
        self.swing_forward = None
        self.active_shift_target = None

    def desired_shift_for_leg(self, leg_name):
        if leg_name in ["FR", "RR"]:
            shift_y = 0.035
        else:
            shift_y = -0.035

        if leg_name in ["FR", "FL"]:
            shift_x = -0.02
        else:
            shift_x = 0.02

        return shift_x, shift_y

    def is_last_sequence_leg(self, leg_name):
        return leg_name == self.step_sequence[-1]

    def uses_conservative_transfer(self, leg_name):
        return leg_name == "RR" or self.is_last_sequence_leg(leg_name)

    def com_shift_duration_for_leg(self, leg_name):
        if self.uses_conservative_transfer(leg_name):
            return self.last_leg_com_shift_duration
        return self.com_shift_duration

    def restabilize_hold_ratio_for_leg(self, leg_name):
        if self.uses_conservative_transfer(leg_name):
            return self.last_leg_restabilize_hold_ratio
        return self.restabilize_hold_ratio

    def step_delta_for_leg(self, leg_name):
        if self.uses_conservative_transfer(leg_name):
            return self.step_delta_x * self.last_leg_step_scale
        return self.step_delta_x

    def compute_support_shift_target(self, swing_leg, support_legs):
        static_shift = np.array(self.desired_shift_for_leg(swing_leg), dtype=float)
        _com, foot_positions = self.get_level_statics()
        foot_positions = np.array(foot_positions)
        all_feet_mean = np.mean(foot_positions, axis=0)
        support_points = np.array([foot_positions[self.leg_index[leg]] for leg in support_legs])
        support_centroid = np.mean(support_points, axis=0)
        dynamic_shift = support_centroid[:2] - all_feet_mean[:2]

        swing_point = foot_positions[self.leg_index[swing_leg]]
        away_from_swing = support_centroid[:2] - swing_point[:2]
        away_norm = np.linalg.norm(away_from_swing)
        if away_norm > 1e-6:
            if self.uses_conservative_transfer(swing_leg):
                margin = min(0.03, 0.35 * away_norm)
            else:
                margin = min(0.015, 0.2 * away_norm)
            dynamic_shift += margin * away_from_swing / away_norm

        if self.uses_conservative_transfer(swing_leg):
            blended_shift = 0.15 * static_shift + 0.85 * dynamic_shift
            blended_shift[0] = np.clip(blended_shift[0], -0.075, 0.075)
            blended_shift[1] = np.clip(blended_shift[1], -0.075, 0.075)
        else:
            blended_shift = 0.35 * static_shift + 0.65 * dynamic_shift
            blended_shift[0] = np.clip(blended_shift[0], -0.06, 0.06)
            blended_shift[1] = np.clip(blended_shift[1], -0.06, 0.06)
        return blended_shift

    def get_swing_phase_target_for_leg(self, leg_name, phase_name, phase_time):
        if self.swing_home is None:
            idx = 3 * self.leg_index[leg_name]
            self.swing_home = self.fk_leg(self.joint_q[idx:idx + 3], leg_name)
            self.swing_lifted = self.swing_home + np.array([0.0, 0.0, self.swing_lift_height])
            self.swing_forward = self.swing_home + np.array([self.step_delta_for_leg(leg_name), 0.0, 0.0])

        if phase_name == "lift":
            alpha = phase_time / self.fr_lift_duration
            return self.interpolate(self.swing_home, self.swing_lifted, alpha)
        if phase_name == "hold":
            return self.swing_lifted
        if phase_name == "step":
            alpha = phase_time / self.fr_step_duration
            target = self.interpolate(self.swing_lifted, self.swing_forward, alpha)
            if alpha < 0.8:
                target[2] = self.swing_lifted[2]
            else:
                touchdown_alpha = (alpha - 0.8) / 0.2
                target[2] = self.interpolate(self.swing_lifted[2], self.swing_forward[2], touchdown_alpha)
            return target
        return None

    def leg_cycle_duration(self, leg_slot):
        restabilize_duration = self.final_stabilize_duration if leg_slot == len(self.step_sequence) - 1 else self.inter_leg_stabilize_duration
        swing_leg = self.step_sequence[leg_slot]
        return (
            self.com_shift_duration_for_leg(swing_leg)
            + self.fr_lift_duration
            + self.fr_hold_duration
            + self.fr_step_duration
            + restabilize_duration
        )
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

    def run(self):
        print("Waiting for state...")
        while not self.state_received:
            time.sleep(0.01)

        start_time = time.perf_counter()
        sequence_start = start_time + self.time_up
        cycle_durations = [self.leg_cycle_duration(i) for i in range(len(self.step_sequence))]
        cycle_boundaries = []
        cycle_sum = self.initial_stabilize_duration
        for duration in cycle_durations:
            cycle_sum += duration
            cycle_boundaries.append(cycle_sum)
        total_sequence_duration = cycle_boundaries[-1]

        while True:
            t = time.perf_counter()
            elapsed = t - start_time

            if elapsed < self.time_up:
                self.announce_phase("stand_up")
                phase = np.tanh(elapsed / 1.5)
                self.command_stand_pose(phase)
            else:
                seq_t = t - sequence_start
                tau_all = np.zeros(12)

                if seq_t < self.initial_stabilize_duration:
                    self.announce_phase("initial_four_leg_stabilize")
                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des
                    tau_all = self.compute_stance_torques(self.leg_names)
                elif seq_t < total_sequence_duration:
                    leg_slot = 0
                    cycle_start = self.initial_stabilize_duration
                    for boundary_idx, boundary in enumerate(cycle_boundaries):
                        if seq_t < boundary:
                            leg_slot = boundary_idx
                            break
                        cycle_start = boundary

                    leg_time = seq_t - cycle_start
                    restabilize_duration = (
                        self.final_stabilize_duration
                        if leg_slot == len(self.step_sequence) - 1
                        else self.inter_leg_stabilize_duration
                    )

                    if self.active_leg_slot != leg_slot:
                        self.reset_swing_targets()
                        self.active_leg_slot = leg_slot

                    swing_leg = self.step_sequence[leg_slot]
                    support_legs = [leg for leg in self.leg_names if leg != swing_leg]
                    if self.active_shift_target is None:
                        self.active_shift_target = self.compute_support_shift_target(swing_leg, support_legs)
                    shift_x, shift_y = self.active_shift_target
                    com_shift_duration = self.com_shift_duration_for_leg(swing_leg)
                    restabilize_hold_ratio = self.restabilize_hold_ratio_for_leg(swing_leg)

                    phase_shift_end = com_shift_duration
                    phase_lift_end = phase_shift_end + self.fr_lift_duration
                    phase_hold_end = phase_lift_end + self.fr_hold_duration
                    phase_step_end = phase_hold_end + self.fr_step_duration

                    if leg_time < phase_shift_end:
                        self.announce_phase(f"{swing_leg}_com_shift")
                        alpha = leg_time / com_shift_duration
                        self.x_des = self.interpolate(self.nominal_x_des, shift_x, alpha)
                        self.y_des = self.interpolate(self.nominal_y_des, shift_y, alpha)
                        tau_all = self.compute_stance_torques(self.leg_names)
                    elif leg_time < phase_lift_end:
                        self.announce_phase(f"{swing_leg}_lift")
                        self.x_des = shift_x
                        self.y_des = shift_y
                        phase_t = leg_time - phase_shift_end
                        tau_all = self.compute_stance_torques(support_legs)
                        swing_target = self.get_swing_phase_target_for_leg(swing_leg, "lift", phase_t)
                        idx = 3 * self.leg_index[swing_leg]
                        tau_all[idx:idx + 3] = self.swing_foot_torque(swing_leg, swing_target)
                    elif leg_time < phase_hold_end:
                        self.announce_phase(f"{swing_leg}_hold")
                        self.x_des = shift_x
                        self.y_des = shift_y
                        tau_all = self.compute_stance_torques(support_legs)
                        swing_target = self.get_swing_phase_target_for_leg(swing_leg, "hold", 0.0)
                        idx = 3 * self.leg_index[swing_leg]
                        tau_all[idx:idx + 3] = self.swing_foot_torque(swing_leg, swing_target)
                    elif leg_time < phase_step_end:
                        self.announce_phase(f"{swing_leg}_step")
                        self.x_des = shift_x
                        self.y_des = shift_y
                        phase_t = leg_time - phase_hold_end
                        tau_all = self.compute_stance_torques(support_legs)
                        swing_target = self.get_swing_phase_target_for_leg(swing_leg, "step", phase_t)
                        idx = 3 * self.leg_index[swing_leg]
                        tau_all[idx:idx + 3] = self.swing_foot_torque(swing_leg, swing_target)
                    else:
                        self.announce_phase(f"{swing_leg}_restabilize")
                        alpha = (leg_time - phase_step_end) / restabilize_duration
                        if alpha < restabilize_hold_ratio:
                            self.x_des = shift_x
                            self.y_des = shift_y
                        else:
                            settle_alpha = (alpha - restabilize_hold_ratio) / (1.0 - restabilize_hold_ratio)
                            self.x_des = self.interpolate(shift_x, self.nominal_x_des, settle_alpha)
                            self.y_des = self.interpolate(shift_y, self.nominal_y_des, settle_alpha)
                        tau_all = self.compute_stance_torques(self.leg_names)
                elif seq_t < total_sequence_duration + self.sequence_done_hold:
                    self.announce_phase("sequence_complete_hold")
                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des
                    tau_all = self.compute_stance_torques(self.leg_names)
                else:
                    self.announce_phase("Sequence completed: four_leg_restabilize")
                    alpha = min(
                        (
                            seq_t
                            - self.initial_stabilize_duration
                            - self.com_shift_duration
                            - self.fr_lift_duration
                            - self.fr_hold_duration
                            - self.fr_step_duration
                        ) / self.final_stabilize_duration,
                        1.0,
                    )
                    self.x_des = self.interpolate(self.shift_x_des, self.nominal_x_des, alpha)
                    self.y_des = self.interpolate(self.shift_y_des, self.nominal_y_des, alpha)
                    tau_all = self.compute_stance_torques(self.leg_names)


                self.apply_tau_command(tau_all)

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)

            loop_elapsed = time.perf_counter() - t
            if loop_elapsed < self.dt:
                time.sleep(self.dt - loop_elapsed)


if __name__ == "__main__":

    sim = True
    args = [arg for arg in sys.argv[1:] if arg != "--sim" and arg != "--real"]
    interface = args[0] if args else None

    if not sim:
        print("WARNING: REAL ROBOT LOW-LEVEL CONTROL.")
        input("Press Enter to continue...")

    controller = Go2FourLegSequenceController(interface=interface, sim=sim)
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

