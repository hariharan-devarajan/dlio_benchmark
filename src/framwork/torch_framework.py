from src.common.enumerations import FormatType
from src.common.error_code import ErrorCodes
from src.framwork.framework import Framework
from src.reader.reader_factory import ReaderFactory


class TorchFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        self.reader_handler = None

    def init_reader(self, format_type):
        if format_type == FormatType.TFRECORD:
            raise Exception(str(ErrorCodes.EC1001))
        self.reader_handler = ReaderFactory.get_format(format_type)


    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
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
