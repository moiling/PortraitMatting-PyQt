from enum import Enum


class TaskState(Enum):
    FAIL         = -1
    INITIAL      = 0
    MATTING      = 1
    MATTING_DONE = 2
    CUTOUT       = 3
    CUTOUT_DONE  = 4
    COMPING      = 5
    COMPING_DONE = 6


class Task:
    orig_path  = ''
    name       = ''
    short_name = ''
    orig_img   = None
    cutout     = None
    fg         = None
    bg         = None
    alpha      = None
    comp       = None
    trimap     = None

    state = TaskState.INITIAL

    def __init__(self, orig_path):
        super().__init__()
        self.orig_path = orig_path
        self.name = orig_path.split('/')[-1].split('\\')[-1]

        self.short_name = self.name

        max_len = 20
        if len(self.short_name) > max_len:
            self.short_name = '...' + self.name[-max_len:]
