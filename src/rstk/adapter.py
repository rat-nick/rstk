class Adapter:
    """
    Base class for all other Adapters.
    Its role is to translate inner and raw IDs, converting input data into the format that the model expects
    and converting output data into the format that the engine should return.
    """

    def convert_input(self, *args, **kwargs):
        """
        The method that should be implemented by all adapters.
        Converts input data into the format that the model expects.
        """
        pass

    def convert_output(self, *args, **kwargs):
        """
        The method that should be implemented by all adapters.
        Converts output data into the format that the engine return.
        """
        pass
