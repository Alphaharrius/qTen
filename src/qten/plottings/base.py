from abc import ABC
import importlib
from typing import (
    Callable,
    Dict,
    ClassVar,
    Tuple,
    Type,
)


class Plottable(ABC):
    """
    An object that supports dynamic plotting backends.
    """

    _registry: ClassVar[Dict[Tuple[Type, str, str], Callable]] = {}
    _backends_loaded: ClassVar[bool] = False

    @classmethod
    def _ensure_backends_loaded(cls) -> None:
        if Plottable._backends_loaded:
            return

        importlib.import_module("qten.plottings._mpl_impl")
        importlib.import_module("qten.plottings._plotly_impl")
        Plottable._backends_loaded = True

    @classmethod
    def register_plot_method(cls, name: str, backend: str = "plotly"):
        """
        Decorator to register a plotting function for a specific class.
        Usage: @MyClass.register_plot_method("scatter")
        """

        def decorator(func: Callable):
            # We register against 'cls' - the class this method was called on.
            Plottable._registry[(cls, name, backend)] = func
            return func

        return decorator

    def plot(self, method: str, backend: str = "plotly", *args, **kwargs):
        """
        Dispatch the plot method to the registered function via MRO.
        """
        Plottable._ensure_backends_loaded()

        # Iterate over the MRO (Method Resolution Order) of the instance
        for class_in_hierarchy in type(self).__mro__:
            key = (class_in_hierarchy, method, backend)

            # Check the central registry
            if key in Plottable._registry:
                plot_func = Plottable._registry[key]
                return plot_func(self, *args, **kwargs)

        # If we reach here, no method was found. Provide a helpful error.
        self._raise_method_not_found(method, backend)

    def _raise_method_not_found(self, method: str, backend: str):
        # Filter available methods to only those relevant to this object (subclasses of valid types)
        available = []
        for reg_cls, reg_name, reg_backend in Plottable._registry:
            if isinstance(self, reg_cls):
                available.append(f"{reg_name} ({reg_backend})")

        msg = (
            f"No plot method '{method}' with backend '{backend}' found for {type(self).__name__}.\n"
            f"Available methods for this object: {', '.join(available) or 'None'}"
        )
        raise ValueError(msg)
