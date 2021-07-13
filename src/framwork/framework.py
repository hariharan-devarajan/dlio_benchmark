from abc import ABC, abstractmethod

from time import sleep


class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass


class Framework(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init_reader(self, format_type):
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def rank(self):
        pass

    @abstractmethod
    def size(self):
        pass


    @abstractmethod
    def start_framework_profiler(self):
        pass

    @abstractmethod
    def stop_framework_profiler(self):
        pass

    @abstractmethod
    def trace_object(self, string, step, r):
        pass

    @abstractmethod
    def checkpoint(self, step_number):
        pass

    def model(epoch, epoch_number, step, computation_time):
        sleep(computation_time)

    @abstractmethod
    def compute(self, epoch_number, step, computation_time):
        pass

    @abstractmethod
    def get_reader(self):
        pass
