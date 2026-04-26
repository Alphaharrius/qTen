from abc import ABC
from importlib import metadata
from typing import (
    Callable,
    Dict,
    ClassVar,
    List,
    Tuple,
    Type,
)


class Plottable(ABC):
    """
    Base class for objects that dispatch plotting calls to registered backends.

    `Plottable` does not implement plotting directly. Instead, backend packages
    register functions under `(object type, method name, backend name)` keys.
    Users call [`plot()`][qten.plottings.Plottable.plot] on the object they want
    to visualize, and the dispatcher finds the matching registered function
    through the object's method-resolution order.

    Repository usage
    ----------------
    The `qten-plots` extension registers Plotly and Matplotlib implementations
    through the `qten.plottings` entry-point group. Those implementations remain
    private backend details; the public user-facing API is the dispatcher call,
    for example `tensor.plot("heatmap", backend="plotly")`.
    """

    _registry: ClassVar[Dict[Tuple[Type, str, str], Callable]] = {}
    _backends_loaded: ClassVar[bool] = False
    _backend_load_errors: ClassVar[List[str]] = []

    @classmethod
    def _ensure_backends_loaded(cls) -> None:
        if Plottable._backends_loaded:
            return

        Plottable._backend_load_errors.clear()
        for entry_point in metadata.entry_points(group="qten.plottings"):
            try:
                entry_point.load()
            except Exception as exc:
                Plottable._backend_load_errors.append(
                    f"{entry_point.name}: {type(exc).__name__}: {exc}"
                )
        Plottable._backends_loaded = True

    @classmethod
    def register_plot_method(cls, name: str, backend: str = "plotly"):
        """
        Register a backend plotting function for this plottable class.

        The returned decorator stores the function in the global plotting
        registry. Registered functions receive the object being plotted as their
        first argument, followed by any extra positional and keyword arguments
        supplied to [`plot()`][qten.plottings.Plottable.plot].

        Parameters
        ----------
        name : str
            User-facing plot method name, such as `scatter`, `structure`, or
            `heatmap`.
        backend : str
            Backend name that selects the implementation. The `qten-plots`
            extension currently uses `plotly` and `matplotlib`.

        Returns
        -------
        Callable
            Decorator that registers the provided plotting function and returns
            it unchanged.
        """

        def decorator(func: Callable):
            # We register against 'cls' - the class this method was called on.
            Plottable._registry[(cls, name, backend)] = func
            return func

        return decorator

    def plot(self, method: str, backend: str = "plotly", *args, **kwargs):
        """
        Dispatch a named plot method to a registered backend implementation.

        The dispatcher first loads plotting entry points, then searches the
        instance type and its base classes for a matching `(type, method,
        backend)` registration. Additional arguments are forwarded unchanged to
        the selected backend function.

        Parameters
        ----------
        method : str
            Plot method name registered for this object's type.
        backend : str
            Backend implementation to use. The `qten-plots` extension currently
            registers `plotly` and `matplotlib`.
        args
            Positional arguments forwarded to the registered plotting function.
        kwargs
            Keyword arguments forwarded to the registered plotting function.

        Returns
        -------
        object
            Backend-specific figure object returned by the registered plotting
            function, such as a Plotly or Matplotlib figure.

        Raises
        ------
        ValueError
            If no plotting function is registered for the requested method and
            backend on this object.

        See Also
        --------
        qten_plots.plottables.PointCloud
            Public plottable helper object provided by the plotting extension.
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
        if Plottable._backend_load_errors:
            msg += "\nExtension load errors: " + "; ".join(
                Plottable._backend_load_errors
            )
        raise ValueError(msg)
