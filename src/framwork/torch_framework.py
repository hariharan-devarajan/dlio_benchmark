from src.common.enumerations import Profiler, FormatType
from src.common.error_code import ErrorCodes
from src.framwork.framework import Framework, DummyTraceObject
from src.profiler.profiler_factory import ProfilerFactory

import torch

from src.utils.argument_parser import ArgumentParser

import horovod.torch as hvd
import os

from src.reader.reader_factory import ReaderFactory

hvd.init()

import functools

HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator

@implements(torch.mean)
def torch_sleep(sleep_time):
    from time import sleep
    return sleep(sleep_time)

class TorchFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        self.profiling = profiling
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
        """
        Barrier implementation using horovod's all-reduce
        """
        const = torch.tensor(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        pass

    def stop_framework_profiler(self):
        pass

    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    def checkpoint(self, step_number):
        """
                Performs Checkpointing for a specific step number. It writes different file of different sizes.
                """
        if not os.path.exists(self.arg_parser.args.output_folder):
            os.makedirs(self.arg_parser.args.output_folder)
        model_file = os.path.join(self.arg_parser.args.output_folder,
                                  "model_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file1 = os.path.join(self.arg_parser.args.output_folder,
                                 "file1_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        bak_file2 = os.path.join(self.arg_parser.args.output_folder,
                                 "file2_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        meta_file = os.path.join(self.arg_parser.args.output_folder,
                                 "meta_{}_{}.bin".format(step_number, self.arg_parser.args.my_rank))
        f = open(model_file, "w")
        string_val = "x" * (1024 * 1024 * 4)
        f.write(string_val)
        f.close()
        f = open(bak_file1, "w")
        string_val = "x" * (1024 * 64)
        f.write(string_val)
        f.close()
        f = open(bak_file2, "w")
        string_val = "x" * (1024 * 4)
        f.write(string_val)
        f.close()
        f = open(meta_file, "w")
        string_val = "x" * (1024)
        f.write(string_val)
        f.close()

    def compute(self, epoch_number, step, computation_time):
        torch_sleep(computation_time)

    def get_reader(self):
        return self.reader_handler
