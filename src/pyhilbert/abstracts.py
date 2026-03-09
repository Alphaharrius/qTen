from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    ClassVar,
    Self,
    Tuple,
    Generic,
    Type,
    TypeVar,
    cast,
)
from typing_extensions import final

from multipledispatch import dispatch


@dataclass(frozen=True)
class Operable(ABC):
    def __add__(self, other):
        return operator_add(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __sub__(self, other):
        return operator_sub(self, other)

    def __mul__(self, other):
        return operator_mul(self, other)

    def __matmul__(self, other):
        return operator_matmul(self, other)

    def __truediv__(self, other):
        return operator_truediv(self, other)

    def __floordiv__(self, other):
        return operator_floordiv(self, other)

    def __pow__(self, other):
        return operator_pow(self, other)

    def __eq__(self, value):
        return operator_eq(self, value)

    def __lt__(self, other):
        return operator_lt(self, other)

    def __le__(self, other):
        return operator_le(self, other)

    def __gt__(self, other):
        return operator_gt(self, other)

    def __ge__(self, other):
        return operator_ge(self, other)

    def __and__(self, other):
        return operator_and(self, other)

    def __or__(self, other):
        return operator_or(self, other)

    def __radd__(self, other):
        return operator_add(other, self)

    def __rsub__(self, other):
        return operator_sub(other, self)

    def __rmul__(self, other):
        return operator_mul(other, self)

    def __rtruediv__(self, other):
        return operator_truediv(other, self)


@dispatch(Operable, Operable)
def operator_add(a, b):
    raise NotImplementedError(f"Addition of {type(a)} and {type(b)} is not supported!")


@dispatch(Operable)
def operator_neg(a):
    raise NotImplementedError(f"Negation of {type(a)} is not supported!")


@dispatch(Operable, Operable)
def operator_sub(a, b):
    return a + (-b)


@dispatch(Operable, Operable)
def operator_mul(a, b):
    raise NotImplementedError(
        f"Multiplication of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_matmul(a, b):
    raise NotImplementedError(
        f"Matrix multiplication of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_truediv(a, b):
    raise NotImplementedError(f"Division of {type(a)} and {type(b)} is not supported!")


@dispatch(Operable, Operable)
def operator_floordiv(a, b):
    raise NotImplementedError(
        f"Floor division of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_pow(a, b):
    raise NotImplementedError(
        f"Exponentiation of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_eq(a, b):
    raise NotImplementedError(
        f"Equality comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_lt(a, b):
    raise NotImplementedError(
        f"Less-than comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_le(a, b):
    raise NotImplementedError(
        f"Less-than-or-equal comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_gt(a, b):
    raise NotImplementedError(
        f"Greater-than comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_ge(a, b):
    raise NotImplementedError(
        f"Greater-than-or-equal comparison of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_and(a, b):
    raise NotImplementedError(
        f"Logical AND of {type(a)} and {type(b)} is not supported!"
    )


@dispatch(Operable, Operable)
def operator_or(a, b):
    raise NotImplementedError(
        f"Logical OR of {type(a)} and {type(b)} is not supported!"
    )


UpdatableType = TypeVar("UpdatableType", bound="Updatable")


class Updatable(ABC, Generic[UpdatableType]):
    """
    An object that can be updated to a new state.
    """

    def update(self, **kwargs) -> UpdatableType:
        """
        Return an updated instance of this object.

        This method delegates the construction of the updated object to
        :meth:`_updated`, then enforces common safety and dataclass consistency
        rules:
        1. ``_updated`` must return a new object (not ``self``).
        2. If both ``self`` and the returned object are dataclasses of the same
           runtime type, fields with ``init=False`` are copied from ``self`` to
           the returned object.

        Parameters
        ----------
        `**kwargs` : `Any`
            Keyword arguments forwarded to :meth:`_updated` to define the new
            state.

        Returns
        -------
        `UpdatableType`
            A new instance representing the updated state.

        Raises
        ------
        `RuntimeError`
            If :meth:`_updated` returns ``self`` instead of a new instance.
        """
        out = self._updated(**kwargs)
        if out is self:
            raise RuntimeError(
                f"{type(self).__name__}._updated() must not return self; return a new object."
            )
        if type(out) is type(self) and is_dataclass(self) and is_dataclass(out):
            for f in fields(self):
                if not f.init:
                    object.__setattr__(out, f.name, getattr(self, f.name))
        return out

    @abstractmethod
    def _updated(self, **kwargs) -> UpdatableType:
        pass


class Plottable(ABC):
    """
    An object that supports dynamic plotting backends.
    """

    _registry: ClassVar[Dict[Tuple[Type, str, str], Callable]] = {}

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


class HasDual(ABC):
    """
    An object that has a dual.
    """

    @property
    @abstractmethod
    def dual(self):
        raise NotImplementedError()


BaseType = TypeVar("BaseType")


class HasBase(Generic[BaseType], ABC):
    """
    An object that is expressed in a specific base (basis/coordinate system).

    "Base" here means the same thing you would mean for a vector: a basis (or
    basis-like structure) that defines how the object's representation is
    written. Examples include a vector's basis, a lattice/affine space's basis,
    a function expanded in a basis of functions, or an operator expressed in a
    particular coordinate frame.

    The key idea is that the *mathematical object* is the same, but its
    *representation* depends on the base. Implementations should therefore
    provide `rebase(...)` to return a new equivalent object expressed in a new
    base, without mutating the original.
    """

    @abstractmethod
    def base(self) -> BaseType:
        """
        Return the base (basis/coordinate system) this object is currently expressed in.

        This should be a lightweight, stable descriptor of the representation context
        (e.g., a basis matrix, lattice, coordinate frame, or function basis). The
        returned base is used by `rebase(...)` to construct an equivalent object in a
        new base, so implementations should not mutate internal state and should
        prefer returning an immutable or effectively immutable object.
        """
        raise NotImplementedError()

    @abstractmethod
    def rebase(self, new_base: BaseType) -> "HasBase[BaseType]":
        """
        Return an equivalent object expressed in ``new_base``.

        Implementations must preserve the underlying mathematical object while
        changing only its representation. This method should be pure: do not
        mutate ``self`` or ``new_base``. Prefer returning a new instance, even if
        the base is unchanged; if you choose to return ``self`` for identical
        bases, document that behavior and ensure immutability.

        ``new_base`` is expected to be compatible with the object. If it is not,
        raise a clear error (typically ``ValueError``). Do not silently coerce
        incompatible bases.
        """
        raise NotImplementedError()


_InnerProductType = TypeVar("_InnerProductType")


class AbstractKet(Generic[_InnerProductType], ABC):
    """
    The base class for all ket-like objects that the inner product is defined via `<bra|ket>` syntax.

    The `_InnerProductType` is the type of the inner product mapping between this ket and its dual bra.
    """

    @abstractmethod
    def ket(self, another: Self) -> _InnerProductType:
        """Return the inner product mapping between this ket and `another` ket."""
        raise NotImplementedError()


class Functional(ABC):
    _registered_methods: ClassVar[Dict[Tuple[type, type], Callable]] = {}
    _resolved_methods: ClassVar[Dict[Tuple[type, type], Callable]] = {}

    @classmethod
    def _invalidate_resolved_methods(cls, obj_type: type) -> None:
        stale_keys = [
            key
            for key in cls._resolved_methods
            if key[1] is cls and issubclass(key[0], obj_type)
        ]
        for key in stale_keys:
            del cls._resolved_methods[key]

    @classmethod
    def register(cls, obj_type: type):
        """
        Register a function defining the action of the `Functional` on a specific object type.
        Dispatch is resolved at call time via MRO, so only the exact `(obj_type, cls)`
        key is stored here.

        Parameters
        ----------
        `obj_type` : `type`
            The type of object the function applies to.
        Returns
        -------
        `Callable`
            A decorator that registers the function for the specified object type.
        """

        def decorator(func: Callable):
            cls._registered_methods[(obj_type, cls)] = func
            cls._invalidate_resolved_methods(obj_type)
            return func

        return decorator

    @classmethod
    def _resolve_method(
        cls, obj_class: type, functional_class: type
    ) -> Callable | None:
        key = (obj_class, functional_class)
        method = cls._resolved_methods.get(key)
        if method is not None:
            return method

        table_get = cls._registered_methods.get
        for obj_super in obj_class.__mro__:
            method = table_get((obj_super, functional_class))
            if method is not None:
                cls._resolved_methods[key] = method
                return method
        return None

    @staticmethod
    def get_applicable_types(cls) -> Tuple[Type, ...]:
        """
        Get all object types that can be applied by this `Functional`.

        Returns
        -------
        Tuple[Type, ...]
            A tuple of all registered object types that this `Functional` can handle.
        """
        types = set()
        for obj_type, functional_type in cls._registered_methods.keys():
            if functional_type is cls:
                types.add(obj_type)
        return tuple(types)

    def allows(self, obj: Any) -> bool:
        """
        Check if this `Functional` can be applied on the given object.

        Parameters
        ----------
        obj : Any
            The object to check for applicability.

        Returns
        -------
        bool
            True if this `Functional` can be applied on the object, False otherwise.
        """
        return self._resolve_method(type(obj), type(self)) is not None

    def invoke(self, obj: Any, **kwargs) -> Any:
        functional_class = type(self)
        obj_class = type(obj)
        method = self._resolve_method(obj_class, functional_class)

        if method is None:
            raise NotImplementedError(
                f"No function registered for {obj_class.__name__} "
                f"with {functional_class.__name__}"
            )

        return method(self, obj, **kwargs)

    def __call__(self, obj: Any, **kwargs) -> Any:
        return self.invoke(obj, **kwargs)


_ElementType = TypeVar("_ElementType")


class Span(ABC, Generic[_ElementType]):
    """
    An object representing the span of a set of elements.

    The specific meaning of "span" depends on the context. For example, in a
    vector space, the span of a set of vectors is the set of all linear
    combinations of those vectors. In a topological space, the span of a set
    of points might be the smallest closed set containing those points.

    Implementations should define what it means to be an element and how to
    determine if an object is contained within the span.

    The `_ElementType` type variable represents the type of elements that define the span.
    """

    @abstractmethod
    def elements(self) -> Tuple[_ElementType, ...]:
        """
        Return the elements contained in this span.

        Returns
        -------
        `Tuple[_ElementType, ...]`
            Immutable tuple of elements represented by this span.
        """
        pass

    def contains(self, v) -> bool:
        """
        Check whether this span contains `v`.

        Supported input
        ---------------
        - `Span`: compared directly via `v.elements()`.
        - `Convertible`: converted to `type(self)` by
          `v.convert(type(self))`, then compared via elements.

        Unsupported input
        -----------------
        - Any non-`Span` object that is not `Convertible`.
        - `Convertible` objects without a registered conversion to `type(self)`.

        Parameters
        ----------
        `v` : `Any`
            Membership query object.

        Returns
        -------
        `bool`
            `True` when all elements in the query span are contained in this span.

        Raises
        ------
        `ValueError`
            If `v` cannot be converted to `type(self)`.

        Developer Note
        --------------
        To make a custom type supported by `contains`, implement `Convertible`
        and register a conversion to the target span type:
        `@YourType.add_conversion(TargetSpanType)`.
        """
        if not isinstance(v, Span):
            if isinstance(v, Convertible):
                v = cast(Convertible, v).convert(type(self))
            else:
                raise ValueError(
                    f"Cannot convert {type(v).__name__} to {type(self).__name__}!"
                )
        base = set(self.elements())
        return all(el in base for el in v.elements())


class HasUnit(ABC):
    """
    An object that has a unit representation.
    """

    @abstractmethod
    def unit(self) -> Self:
        """Return the unit representation of this object."""
        raise NotImplementedError()


A = TypeVar("A", bound="Convertible")
B = TypeVar("B")
_type_conversion_table: Dict[Tuple[Type[Any], Type[Any]], Callable[[Any], Any]] = {}


class Convertible(ABC):
    """
    Mixin for objects that support explicit type-to-type conversion.

    Conversion functions are registered globally using
    ``@MyType.add_conversion(TargetType)`` with
    ``(source_type, destination_type)`` as the lookup key. Implementers inherit
    :meth:`convert` and usually only need to register conversion handlers.

    Notes
    -----
    Lookup first checks ``(type(self), T)`` exactly. If not found, it scans
    source supertypes in MRO order from immediate parent to the most abstract
    parent. If no conversion function is found, conversion fails with
    ``NotImplementedError``.
    """

    @classmethod
    def add_conversion(
        cls: Type[A], T: Type[B]
    ) -> Callable[[Callable[[A], B]], Callable[[A], B]]:
        """
        Register a conversion from ``cls`` to ``T``.

        Example
        -------
        `@MyType.add_conversion(TargetType)`
        `def to_target(x: MyType) -> TargetType: ...`
        """

        def decorator(func: Callable[[A], B]) -> Callable[[A], B]:
            _type_conversion_table[(cls, T)] = cast(Callable[[Any], Any], func)
            return func

        return decorator

    @final
    def convert(self, T: Type[B]) -> B:
        """
        Convert this instance to the requested target type.

        Parameters
        ----------
        `T` : `Type[B]`
            Destination type to convert into.

        Returns
        -------
        `B`
            Converted object produced by the registered conversion function.

        Raises
        ------
        `NotImplementedError`
            If no conversion function has been registered for
            ``(type(self), T)`` or any source supertype via
            :meth:`add_conversion`.
        """
        source_type = type(self)
        table_get = _type_conversion_table.get

        convertor = table_get((source_type, T))
        if convertor is None:
            for super_type in source_type.__mro__[1:]:
                convertor = table_get((super_type, T))
                if convertor is not None:
                    # Cache resolved parent conversion under the concrete source type.
                    _type_conversion_table[(source_type, T)] = convertor
                    break

        if convertor is None:
            raise NotImplementedError(
                f"No conversion from {source_type.__name__} to {T.__name__}!"
            )
        return cast(Callable[["Convertible"], B], convertor)(self)
