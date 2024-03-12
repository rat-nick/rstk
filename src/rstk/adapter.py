"""
Module containing the base class for adapters.
Also contains definitions for the most commonly used adapters.
"""


class Adapter:
    """
    Base class for all other Adapters.
    Defines methods for converting input data into the format that the model expects
    and converting output data into the format that the engine should return.
    """

    def __init__(self):
        pass

    def convert_input(self, input, *args, **kwargs):
        """
        The method that should be implemented by all adapters.
        Converts input data into the format that the model expects.
        """
        return input  # pragma: no cover

    def convert_output(self, output, *args, **kwargs):
        """
        The method that should be implemented by all adapters.
        Converts output data into the format that the engine returns.
        """
        return output  # pragma: no cover
