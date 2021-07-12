from abc import ABC, abstractmethod

from time import sleep


class Framework(ABC):
    def __init__(self):
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
    def checkpoint(self, step_number):
        pass

    def model(epoch, step, time):
        sleep(time)

    @abstractmethod
    def compute(self, epoch_number, step, computation_time):
        pass

    @abstractmethod
    def get_reader(self):
        pass
