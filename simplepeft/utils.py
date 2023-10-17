from enum import Enum
import platform

os = platform.system()

IS_WINDOWS = os == "Windows"


class Tasks(Enum):
    ASR = "ASR"
    Text2Text = "T2T"
    TEXT_GEN = "TEXT_GEN"
