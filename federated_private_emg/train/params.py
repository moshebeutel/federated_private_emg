from dataclasses import dataclass


@dataclass
class TrainParams:
    epochs: int
    batch_size: int
    descent_every: int
    validation_every: int
    test_at_end: bool

    def __init__(self, epochs, batch_size, descent_every=1, validation_every=1, test_at_end=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.descent_every= descent_every
        self.validation_every = validation_every
        self.test_at_end = test_at_end

