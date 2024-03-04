class Model:
    """
    Base class for all models to be used for recommendation.
    Their sole purpose is to perform a forward pass with given data.
    """

    def forward(self, *args, **kwargs):
        """
        Method that should be implemented by all models.
        Performs a forward pass with the provided data.
        """
        pass
