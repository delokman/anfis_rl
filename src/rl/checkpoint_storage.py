class LowestCheckpoint:
    """
    Helper class meant to store the locations of the latest checkpoint and be able
    to reload a model
    """

    def __init__(self) -> None:
        super().__init__()

        self.error = None
        self.checkpoint_location = None

    def update(self, error, location):
        """
        Updates the checkpoint, only saves the one with the lowest error

        :param error: model score to update the model on
        :param location: the checkpoint saved location
        """
        if self.error is None or self.error > error:
            self.error = error
            self.checkpoint_location = location

    def reload(self, model):
        """
        Call the `load_checkpoint` function on the model class with as parameter
        the best checkpoint location

        :param model: the model to reload
        """
        model.load_checkpoint(self.checkpoint_location)
