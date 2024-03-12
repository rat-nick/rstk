"""
Module containing the base class for adapters.
Also contains definitions for the most commonly used adapters.
"""

from typing import Callable, Iterable

from .dataset import Dataset, FeatureDataset


def identity(x):
    """
    A simple function that returns the input value as is.
    """
    if isinstance(x, Iterable) and not isinstance(x, str):
        return type(x)(identity(item) for item in x)
    else:
        return x


class Adapter:
    """
    Base class for all other Adapters.
    Defines methods for converting input data into the format that the model expects
    and converting output data into the format that the engine should return.
    The base Adapter does nothing with the input.
    """

    def __init__(
        self,
        function: Callable = identity,
        inverse_function: Callable = identity,
        next: "Adapter | None" = None,
    ):
        self.function = function
        self.inverse_function = inverse_function
        self.next = next

    def __call__(self, input, *args, **kwargs):
        return self.forward(input, *args, **kwargs)

    def forward(self, input, *args, **kwargs):
        """
        Perform adaptation on the input using the adapter_function and any other nested Adapters.
        Args:
            input: The input data to be adapted.
        Returns:
            The adapted input data.
        """
        input = self.function(input)

        if self.next is not None:
            input = self.next.forward(input, *args, **kwargs)

        return input

    def backward(self, input, *args, **kwargs):
        """
        Perform inverse adaptation **if possible** on the input using the inverse_function and any other nested Adapters.
        Args:
            input: The input data to be adapted.
        Returns:
            The adapted input data.
        """
        if self.inverse_function is None:
            raise NotImplementedError(
                "Inverse function is not applicable for this adapter."
            )

        if self.next is not None:
            input = self.next.backward(input, *args, **kwargs)

        input = self.inverse_function(input)

        return input

    @property
    def inverted(self) -> "Adapter":
        """
        Returns the inverse adapter of this adapter.
        """
        return Adapter(
            function=self.backward,
            inverse_function=self.forward,
            next=self.next,
        )


class IDAdapter(Adapter):
    """
    Converts raw IDs to inner IDs.
    """

    def __init__(
        self,
        dataset: Dataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.function = lambda x: self.dataset.raw2inner[x]
        self.inverse_function = lambda x: self.dataset.inner2raw[x]

    def forward(self, input, *args, **kwargs):
        """
        Converts raw IDs to inner IDs.
        """
        return super().forward(input, *args, **kwargs)

    def backward(self, input, *args, **kwargs):
        """
        Converts inner IDs to raw IDs.
        """
        return super().backward(input, *args, **kwargs)


class FeatureAdapter(Adapter):
    """
    Converts inner IDs to vectors in feature space.
    """

    def __init__(
        self,
        dataset: FeatureDataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.function = lambda x: self.dataset[x]
        self.inverse_function = None


class AdapterBuilder:
    """
    Class for building composite adapters.
    """

    def __init__(self) -> None:
        self.product = None

    def _reset(self):
        self.product = None

    def add(
        self,
        adapter: Adapter,
    ) -> "AdapterBuilder":
        """
        Add a new adapter to the AdapterBuilder.

        Args:
            adapter (Adapter): The adapter to be added.

        Returns:
            AdapterBuilder: The updated AdapterBuilder.
        """
        if self.product is None:
            self.product = adapter
        else:
            self.product.next = adapter
        return self

    def build(self) -> Adapter:
        """
        Build the adapter and return it.

        :return: Adapter
        """
        ret = self.product
        self._reset()
        if ret is None:
            raise RuntimeError("No adapters were added to the builder.")
        return ret
