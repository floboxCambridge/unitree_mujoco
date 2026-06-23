from paper_four_leg_sequence import Go2FourLegSequenceController
import time

class Go2TwoLegSequenceController(Go2FourLegSequenceController):
    def __init__(self, interface=None, sim=True):
        super().__init__(interface=interface, sim=sim)
        self.step_sequence = ["FR", "FL"]
        self.final_stabilize_duration = 2.0
        self.sequence_done_hold = 2.0
        self.active_leg_slot = None
        self.reset_swing_targets()


if __name__ == "__main__":
    import sys

    sim = "--real" not in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--sim" and arg != "--real"]
    interface = args[0] if args else None

    if not sim:
        print("WARNING: REAL ROBOT LOW-LEVEL CONTROL.")
        input("Press Enter to continue...")

    controller = Go2TwoLegSequenceController(interface=interface, sim=sim)
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
