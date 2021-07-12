from src.common.enumerations import FrameworkType
from src.common.error_code import ErrorCodes
from src.framwork.tf_framework import TFFramework
from src.framwork.torch_framework import TorchFramework


class FrameworkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_framework(framework_type, profiling, format_type):
        if framework_type == FrameworkType.TENSORFLOW:
            return TFFramework.get_instance(profiling, format_type)
        elif framework_type == FrameworkType.PYTORCH:
            return TorchFramework.get_instance(profiling, format_type)
        else:
            raise Exception(str(ErrorCodes.EC1001))