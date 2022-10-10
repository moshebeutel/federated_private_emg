from dataclasses import dataclass


@dataclass
class DpParams:
    dp_lot: int
    dp_sigma: float
    dp_C: float

    def __init__(self, dp_lot, dp_sigma, dp_C):
        self.dp_lot = dp_lot
        self.dp_sigma = dp_sigma
        self.dp_C = dp_C


