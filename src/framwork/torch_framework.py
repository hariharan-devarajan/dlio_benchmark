from src.framwork.framework import Framework


class TorchFramework(Framework):
    __instance = None

    def __init__(self, profiling, format_type):
        pass

    @staticmethod
    def get_instance(profiling, format_type):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework(profiling, format_type)
        return TorchFramework.__instance

    def barrier(self):
        pass

    def rank(self):
        pass

    def size(self):
        pass

    def start_framework_profiler(self):
        pass

    def stop_framework_profiler(self):
        pass

    def checkpoint(self, step_number):
        pass

    def compute(self, epoch_number, step, computation_time):
        pass

    def get_reader(self):
        pass
