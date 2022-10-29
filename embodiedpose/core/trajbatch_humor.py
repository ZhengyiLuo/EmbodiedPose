from embodiedpose.core import TrajBatch
import numpy as np


class TrajBatchHumor(TrajBatch):
    def __init__(self, memory_list):
        super().__init__(memory_list)
        self.v_metas = np.stack(next(self.batch))

        self.humor_target = np.stack(next(self.batch))
        self.sim_humor_state = np.stack(next(self.batch))