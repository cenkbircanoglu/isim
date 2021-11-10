import numpy as np
import torch
from torch.utils.data import Subset

from utils import log_loss_summary


class PolyOptimizer(torch.optim.SGD):
    def __init__(
        self,
        params,
        lr,
        weight_decay,
        max_step,
        momentum=0.9,
        nesterov=False,
        logger=None,
    ):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group["lr"] for group in self.param_groups]
        self.logger = logger

    def step(self, closure=None, epoch=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__initial_lr[i] * lr_mult
                if self.logger is not None:
                    log_loss_summary(
                        self.logger,
                        self.param_groups[i]["lr"],
                        epoch,
                        tag=f"lr-group-{i}",
                    )
        # else:
        #     for i in range(len(self.param_groups)):
        #         self.param_groups[i]["lr"] = 1e-4
        super().step(closure)

        self.global_step += 1


def split_dataset(dataset, n_splits):
    return [
        Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)
    ]


if __name__ == "__main__":
    max_step = (10510 // 16) * 3

    optimizer = PolyOptimizer(
        [{"params": dict(), "lr": 0.1, "weight_decay": 1e-4}],
        lr=0.1,
        weight_decay=1e-4,
        max_step=max_step,
    )
    print(max_step)
    for i in range(max_step):
        optimizer.step()
