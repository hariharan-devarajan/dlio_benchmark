from src.common.enumerations import Profiler, FormatType
from src.common.error_code import ErrorCodes
from src.framwork.framework import Framework
from src.profiler.profiler_factory import ProfilerFactory

import tensorflow as tf

from src.utils.argument_parser import ArgumentParser

print(tf.sysconfig.get_link_flags())
import horovod.tensorflow as hvd
import os

from src.reader.reader_factory import ReaderFactory

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()


class TFFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        self.profiling = profiling
        if profiling:
            self.tensorboard = ProfilerFactory.get_profiler(Profiler.TENSORBOARD)
        self.reader_handler = None

    def init_reader(self, format_type):
        if format_type == FormatType.DATA_LOADER:
            raise Exception(str(ErrorCodes.EC1001))
        self.reader_handler = ReaderFactory.get_format(format_type)

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(profiling)
        return TFFramework.__instance

    def barrier(self):
        """
        Barrier implementation using horovod's all-reduce
        """
        const = tf.constant(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    def stop_framework_profiler(self):
        if self.profiling:
            self.tensorboard.stop()

    def trace_object(self, string, step, r):
        return tf.profiler.experimental.Trace(string, step_num=step, _r=r)

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
        tf.function(self.model)(epoch_number, step, computation_time)

    def get_reader(self):
        return self.reader_handler
