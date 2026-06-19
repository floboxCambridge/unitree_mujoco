import time

import numpy as np

from paper_fr_forward_step import Go2FRForwardStepController


class Go2FourLegSequenceController(Go2FRForwardStepController):
    def __init__(self, interface=None, sim=True):
        super().__init__(interface=interface, sim=sim)
        self.step_sequence = ["FR", "FL", "RL", "RR"]
        self.swing_lift_height = 0.07
        self.initial_stabilize_duration = 2.5
        self.inter_leg_stabilize_duration = 1.5
        self.final_stabilize_duration = 2.0
        self.sequence_done_hold = 3.0

        self.swing_home = None
        self.swing_lifted = None
        self.swing_forward = None
        self.active_leg_slot = None

    def reset_swing_targets(self):
        self.swing_home = None
        self.swing_lifted = None
        self.swing_forward = None

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

    def get_swing_phase_target_for_leg(self, leg_name, phase_name, phase_time):
        if self.swing_home is None:
            idx = 3 * self.leg_index[leg_name]
            self.swing_home = self.fk_leg(self.joint_q[idx:idx + 3], leg_name)
            self.swing_lifted = self.swing_home + [0.0, 0.0, self.swing_lift_height]
            self.swing_forward = self.swing_home + [self.step_delta_x, 0.0, 0.0]

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
        return (
            self.com_shift_duration
            + self.fr_lift_duration
            + self.fr_hold_duration
            + self.fr_step_duration
            + restabilize_duration
        )

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
                    shift_x, shift_y = self.desired_shift_for_leg(swing_leg)

                    phase_shift_end = self.com_shift_duration
                    phase_lift_end = phase_shift_end + self.fr_lift_duration
                    phase_hold_end = phase_lift_end + self.fr_hold_duration
                    phase_step_end = phase_hold_end + self.fr_step_duration

                    if leg_time < phase_shift_end:
                        self.announce_phase(f"{swing_leg}_com_shift")
                        alpha = leg_time / self.com_shift_duration
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
                        self.x_des = self.interpolate(shift_x, self.nominal_x_des, alpha)
                        self.y_des = self.interpolate(shift_y, self.nominal_y_des, alpha)
                        tau_all = self.compute_stance_torques(self.leg_names)
                elif seq_t < total_sequence_duration + self.sequence_done_hold:
                    self.announce_phase("sequence_complete_hold")
                    self.x_des = self.nominal_x_des
                    self.y_des = self.nominal_y_des
                    tau_all = self.compute_stance_torques(self.leg_names)
                else:
                    print("Completed four-leg sequence.")
                    return

                self.apply_tau_command(tau_all)

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)

            loop_elapsed = time.perf_counter() - t
            if loop_elapsed < self.dt:
                time.sleep(self.dt - loop_elapsed)


if __name__ == "__main__":
    import sys

    sim = "--real" not in sys.argv
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
