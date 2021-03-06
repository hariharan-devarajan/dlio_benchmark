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

import h5py
from numpy import random
import math

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator
from src.utils.utility import progress
from shutil import copyfile

"""
Generator for creating data in HDF5 format.
"""
class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.chunk_size = self._arg_parser.args.chunk_size
        self.enable_chunking = self._arg_parser.args.enable_chunking

    def generate(self):
        """
        Generate hdf5 data for training. It generates a 3d dataset and writes it to file.
        """
        super().generate()
        samples_per_iter=1024*100
        records = random.random((samples_per_iter, self._dimension, self._dimension))
        record_labels = [0] * self.num_samples
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating HDF5 Data")
                out_path_spec = "{}_{}_of_{}.h5".format(self._file_prefix, i+1, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    hf = h5py.File(out_path_spec, 'w')
                    chunks = None
                    if self.enable_chunking:
                        chunk_dimension = int(math.ceil(math.sqrt(self.chunk_size)))
                        if chunk_dimension > self._dimension:
                            chunk_dimension = self._dimension
                        chunks = (1, chunk_dimension, chunk_dimension)
                    compression = None
                    compression_level = None
                    if self.compression != Compression.NONE:
                        compression = str(self.compression)
                        if self.compression == Compression.GZIP:
                            compression_level = self.compression_level
                    dset = hf.create_dataset('records', (self.num_samples,self._dimension, self._dimension), chunks=chunks, compression=compression,
                                             compression_opts=compression_level)
                    samples_written = 0
                    while samples_written < self.num_samples:
                        if samples_per_iter < self.num_samples-samples_written:
                            samples_to_write = samples_per_iter
                        else:
                            samples_to_write = self.num_samples-samples_written
                        dset[samples_written:samples_written+samples_to_write] = records[:samples_to_write]
                        samples_written += samples_to_write
                    hf.create_dataset('labels', data=record_labels)
                    hf.close()
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)
