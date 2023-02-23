from abc import ABC
import torch


class Scaler(ABC):
    def __init__(self, init_scale):
        self.init_scale = init_scale

    def scale(self, loss):
        return loss * self.init_scale

    def step(self, optimizer):
        pass

    def update(self):
        pass


class StaticScaler(Scaler):
    def __init__(self, init_scale=65536):
        super().__init__(init_scale)

    def step(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not param.isfinite().all():
                    return
                else:
                    param.grad /= self.init_scale
        optimizer.step()


class DynamicScaler(Scaler):
    def __init__(
            self,
            init_scale=65536 * 4,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=10
    ):
        super().__init__(init_scale)
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.good_steps = 0

    def step(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not param.isfinite().all():
                    self.init_scale *= self.backoff_factor
                    print(f"reduce scale {self.init_scale}")
                    self.good_steps = 0
                    optimizer.zero_grad()
                    return
                else:
                    param.grad /= self.init_scale
        self.good_steps += 1
        optimizer.step()
        optimizer.zero_grad()

    def update(self):
        if self.good_steps == self.growth_interval:
            self.init_scale *= self.growth_factor
            self.good_steps = 0
            print(f"grow scale {self.init_scale}")
