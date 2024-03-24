"""
Module containing the base class for recommender engines.
Also contains definitions for the most commonly used engines.
"""

from abc import ABC
from typing import Any, List, Type

from .data.dataset import Dataset, FeatureDataset, UtilityMatrix
from .data.types import ConversionRegistry, InteractionMatrix, Recommendation
from .model import Model, SimilarityBased


class Engine(ABC):
    """
    Base class for recommender systems.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        input_type: Type = InteractionMatrix,
        output_type: Type = Recommendation,
    ) -> None:
        """
        Base class for all recommender engines.
        Initializes the engine with the given dataset, model, input adapter, and output adapter.

        Args:
            dataset (Dataset): The dataset to be used.
            model (Model): The model to be used.
            input_adapter (Type[Adapter] | Adapter, optional): The input adapter to be used. Defaults to Adapter.
            output_adapter (Type[Adapter] | Adapter, optional): The input adapter to be used. Defaults to Adapter.

        Raises:
            ValueError: When either input or output adapter are not supplied as types or instances.

        Note:
            ``input_adapter`` and ``output_adapter`` parameters can be supplied as types or as instances.
        """

        self.dataset = dataset
        self.model = model
        self.input_type = input_type
        self.output_type = output_type

    def recommend(self, input: Any) -> List:
        """
        Retrieves recommendations based on the provided arguments.
        Every recommender system must implement this method.
        """
        input = ConversionRegistry.convert(
            self.input_type, self.model.input_type, input, self.dataset
        )
        output = self.model.forward(input)
        output = ConversionRegistry.convert(
            self.model.output_type, self.output_type, output, self.dataset
        )
        return output


class CBSEngine(Engine):
    """
    Class that implements the Content-Based Similarity Engine
    The items are represented in feature space and the engine
    calculates the similarity between items using the given metric.
    """

    def __init__(
        self,
        dataset: FeatureDataset,
        model: SimilarityBased,
        *args,
        **kwargs,
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
        super().__init__(dataset, model, *args, **kwargs)

        self.model.fit(dataset)


class CFSEngine(Engine):
    """
    Class that implements the Collaborative-Filtering Similarity Engine.
    The items features are the user ratings.

    """

    def __init__(
        self,
        dataset: UtilityMatrix,
        model: SimilarityBased,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the engine with the given dataset, model, input adapter, and output adapter.

        Args:
            dataset (UtilityMatrix): The dataset to be used.
            model (SimilarityBased): The model to be used.
            input_adapter (Type[Adapter] | Adapter, optional): The input adapter to be used. Defaults to Adapter.
            output_adapter (Type[Adapter] | Adapter, optional): The output adapter to be used. Defaults to Adapter.
        Returns:
            None
        """

        super().__init__(dataset, model, *args, **kwargs)

        self.model.fit(dataset)
