from typing import List

from .adapter import Adapter
from .model import Model


class Engine:
    """
    Base interface for recommender systems.
    """

    def __init__(self, model: Model, adapter: Adapter) -> None:
        self.model = model
        self.adapter = adapter

    def get_recommendations(self, *args, **kwargs) -> List:
        """
        Retrieves recommendations based on the provided arguments.
        Every recmommender system must implement this method.
        """
        input = self.adapter.convert_input(*args, **kwargs)
        output = self.model.forward(input)
        return self.adapter.convert_output(output, *args, **kwargs)
