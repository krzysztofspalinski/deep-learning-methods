"""
Optimizer
"""


class Optimizer:
    def __init__(self):
        pass

    def optimize(self, old_weights, new_weights):
        return new_weights


class GDwithMomentum(Optimizer):
    def __init__(self, momentum_ratio=0.9):
        super().__init__()
        self.momentum_ratio = momentum_ratio

    def optimize(self, old_weights, new_weights):
        return old_weights * self.momentum_ratio + \
               new_weights * (1 - self.momentum_ratio)


def main():
    pass


if __name__ == "__main__":
    main()
