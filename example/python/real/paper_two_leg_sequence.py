from paper_four_leg_sequence import Go2FourLegSequenceController


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
