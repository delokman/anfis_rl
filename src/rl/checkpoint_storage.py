import torch


class LowestCheckpoint:
    def __init__(self) -> None:
        super().__init__()

        self.error = None
        self.checkpoint_location = None

    def update(self, error, location):
        if self.error is None or self.error > error:
            self.error = error
            self.checkpoint_location = location

    def reload(self, model):
        model.load_checkpoint(self.checkpoint_location)
