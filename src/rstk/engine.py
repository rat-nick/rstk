"""
Module containing the base class for recommender engines.
Also contains definitions for the most commonly used engines.
"""

from abc import ABC
from typing import List, Type

from .adapter import Adapter, FeatureAdapter
from .dataset import Dataset, FeatureDataset, UtilityMatrix
from .model import Model, SimilarityBased


class Engine(ABC):
    """
    Base class for recommender systems.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        input_adapter: Type[Adapter] | Adapter = Adapter,
        output_adapter: Type[Adapter] | Adapter = Adapter,
        *args,
        **kwargs,
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
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    def recommend(self, input, *args, **kwargs) -> List:
        """
        Retrieves recommendations based on the provided arguments.
        Every recommender system must implement this method.
        """
        input = self.input_adapter.forward(input, *args, **kwargs)
        output = self.model.forward(input)
        return self.output_adapter.forward(output, *args, **kwargs)


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
        input_adapter: Type[Adapter] | Adapter = FeatureAdapter,
        output_adapter: Type[Adapter] | Adapter = Adapter,
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
        super().__init__(dataset, model, input_adapter, output_adapter, *args, **kwargs)

        if type(self.input_adapter) is type:
            self.input_adapter = input_adapter(dataset, *args, **kwargs)

        if type(self.output_adapter) is type:
            self.output_adapter = output_adapter(dataset, *args, **kwargs)

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
        input_adapter: Type[Adapter] | Adapter = Adapter,
        output_adapter: Type[Adapter] | Adapter = Adapter,
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

        super().__init__(dataset, model, input_adapter, output_adapter, *args, **kwargs)

        if type(self.input_adapter) is type:
            self.input_adapter = input_adapter(dataset, *args, **kwargs)

        if type(self.output_adapter) is type:
            self.output_adapter = output_adapter(dataset, *args, **kwargs)

        self.model.fit(dataset)
