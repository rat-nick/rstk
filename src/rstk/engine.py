"""
    Module containing the base class for recommender systems.
    Also contains definitions for the most commonly used engines.
"""

from typing import List

from .adapter import Adapter
from .dataset import Dataset, ItemDataset, UtilityMatrix
from .model import Model, SimilarityBased


class Engine:
    """
    Base class for recommender systems.
    """

    def __init__(
        self, dataset: Dataset, model: Model, adapter: Adapter = Adapter()
    ) -> None:
        """
        Initializes the class with the provided dataset, model, and adapter.

        Parameters:
            dataset (Dataset): The dataset to be used for initialization.
            model (Model): The model to be used for initialization.
            adapter (Adapter): The adapter to be used for initialization.

        """
        self.dataset = dataset
        self.model = model
        self.adapter = adapter

    def get_recommendations(self, *args, **kwargs) -> List:
        """
        Retrieves recommendations based on the provided arguments.
        Every recommender system must implement this method.
        """
        input = self.adapter.convert_input(*args, **kwargs)
        output = self.model.forward(input)
        return self.adapter.convert_output(output, *args, **kwargs)


class CBSEngine(Engine):
    """
    Class that implements the Content-Based Similarity Engine
    The items are represented in feature space and the engine
    calculates the similarity between items using the given metric
    """

    def __init__(
        self,
        dataset: ItemDataset,
        model: SimilarityBased,
        adapter: Adapter,
    ) -> None:
        """
        Initializes the object with the given dataset, model, and adapter.

        Parameters:
            dataset (ItemDataset): The dataset to be used.
            model (SimilarityBased): The similarity based model to be used.
            adapter (Adapter): The adapter to be used.

        Returns:
            None
        """
        super().__init__(dataset, model, adapter)
        self.model.fit(dataset)


class CFBSEngine(Engine):
    """
    Class that implements the Collaborative-Filtering Based Similarity Engine.
    The items features are the user ratings.

    """

    def __init__(
        self,
        dataset: UtilityMatrix,
        model: SimilarityBased,
        adapter: Adapter,
    ) -> None:
        """
        Initialize the object with the given dataset, model, and adapter.

        Args:
            dataset (UtilityMatrix): A utility matrix of user-item interactions.
            model (SimilarityBased): The similarity based model to be used.
            adapter (Adapter): The adapter to be used.

        Returns:
            None
        """
        super().__init__(dataset, model, adapter)
        self.model.fit(dataset)
