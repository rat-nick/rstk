from typing import List


class Recommender:
    """
    Base interface for recommender systems.
    """

    def get_recommendations(self, *args, **kwargs) -> List:
        """
        Retrieves recommendations based on the provided arguments.
        Every recmommender system must implement this method.
        """

        pass
