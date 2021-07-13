"""
 Copyright (C) 2020  Argonne, Hariharan Devarajan <hdevarajan@anl.gov>
 This file is part of DLProfile
 DLIO is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the published by the Free Software Foundation, either
 version 3 of the License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU General Public License along with this program.
 If not, see <http://www.gnu.org/licenses/>.
"""

from src.common.enumerations import Profiler
from src.data_generator.generator_factory import GeneratorFactory
from src.framwork.framework_factory import FrameworkFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.utils.argument_parser import ArgumentParser

import math
import shutil
from time import time
import os


class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        self.arg_parser = ArgumentParser.get_instance()
        self.framework = FrameworkFactory().get_framework(self.arg_parser.args.framework,
                                                          self.arg_parser.args.profiling)
        self.arg_parser.args.my_rank = self.framework.rank()
        self.arg_parser.args.comm_size = self.framework.size()
        self.framework.init_reader(self.arg_parser.args.format)
        self.darshan = None
        self.data_generator = None
        self.my_rank = self.arg_parser.args.my_rank
        self.comm_size = self.arg_parser.args.comm_size
        self.num_files = self.arg_parser.args.num_files
        self.num_samples = self.arg_parser.args.num_samples
        self.batch_size = self.arg_parser.args.batch_size
        self.computation_time = self.arg_parser.args.computation_time

        if self.arg_parser.args.profiling:
            self.darshan = ProfilerFactory().get_profiler(Profiler.DARSHAN)
        if self.arg_parser.args.generate_data:
            self.data_generator = GeneratorFactory.get_generator(self.arg_parser.args.format)

    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        if self.arg_parser.args.debug and self.arg_parser.args.my_rank == 0:
            input("Press enter to start\n")
        self.framework.barrier()
        if self.arg_parser.args.generate_data:
            self.data_generator.generate()
            print("Generation done")
        if self.arg_parser.args.profiling:
            self.darshan.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.arg_parser.args.my_rank == 0:
                print("profiling started")
        self.framework.barrier()

    def _train(self, epoch_number):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        step = 1
        total = math.ceil(self.num_samples * self.num_files / self.batch_size / self.comm_size)
        for element in self.framework.get_reader().next():
            if self.computation_time > 0:
                self.framework.compute(epoch_number, step, self.computation_time)
            if self.arg_parser.args.checkpoint and step % self.arg_parser.args.steps_checkpoint == 0:
                self.framework.checkpoint(step)
            step += 1
            if step > total:
                return step - 1
        return step - 1

    def run(self):
        """
        Run the total epochs for training. On each epoch, it prepares dataset for reading, it trains, and finalizes the
        dataset.
        """
        if not self.arg_parser.args.generate_only:
            for epoch_number in range(0, self.arg_parser.args.epochs):
                start_time = time()
                self.framework.get_reader().read(epoch_number)
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    print("Datasets loaded in {} epochs for rank {} in {} seconds".format(epoch_number + 1,
                                                                                          self.arg_parser.args.my_rank,
                                                                                          (time() - start_time)))
                start_time = time()
                steps = self._train(epoch_number)
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    print("Finished {} steps in {} epochs for rank {}  in {} seconds".format(steps, epoch_number + 1,
                                                                                             self.arg_parser.args.my_rank,
                                                                                             (time() - start_time)))
                self.framework.get_reader().finalize()

    def finalize(self):
        """
        It finalizes the dataset once the training epoch is completed.
        """
        print("Finalizing for rank {}".format(self.arg_parser.args.my_rank))
        self.framework.barrier()
        if not self.arg_parser.args.generate_only:
            if self.arg_parser.args.profiling:
                self.darshan.stop()
                self.framework.stop_framework_profiler.stop()
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    print("profiling stopped")
            if not self.arg_parser.args.keep_files:
                self.framework.barrier()
                if self.arg_parser.args.my_rank == 0:
                    if os.path.exists(self.arg_parser.args.data_folder):
                        shutil.rmtree(self.arg_parser.args.data_folder)
                        print("Deleted data files")
        #


if __name__ == '__main__':
    """
    The main method to start the benchmark runtime.
    """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark()
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    exit(0)
