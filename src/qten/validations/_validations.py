"""
Runtime validation decorators for QTen dataclasses.

The validation layer is intentionally small: classes opt in with
[`need_validation()`][qten.validations.need_validation], validators are plain
callables that receive the constructed instance, and
[`validate()`][qten.validations.validate] can rerun the configured chain on
an existing value. [`no_validate`][qten.validations.no_validate] provides a
thread-local escape hatch for construction paths that need to create temporary
or intentionally incomplete objects.

Repository usage
----------------
QTen symbolic spaces, geometry objects, and tensor containers use these helpers
to keep dataclass normalization and validation close to the type definitions
without requiring each class to reimplement `__post_init__` plumbing.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from contextlib import ContextDecorator
from functools import wraps
from threading import local
from typing import Any, Callable, List, Type, TypeVar, cast


_VALIDATION_STATE = local()

_C = TypeVar("_C", bound=Type[Any])
ValidatorFn = Callable[[Any], None]


def _validation_disabled_depth() -> int:
    return cast(int, getattr(_VALIDATION_STATE, "disabled_depth", 0))


def _class_validators(cls: type[Any]) -> List[ValidatorFn]:
    return cast(List[ValidatorFn], getattr(cls, "__validators__", ()))


class no_validate(ContextDecorator):
    """
    Temporarily disable validation in the current thread.

    While this context manager is active, construction-time validators installed
    by [`need_validation()`][qten.validations.need_validation] do not run. The
    suppression state is thread-local and nestable, so entering nested
    `no_validate()` blocks only reenables validation after the outermost block
    exits.

    Because this class inherits from `contextlib.ContextDecorator`, it can also
    be used as a function decorator around construction helper functions.

    Examples
    --------
    ```python
    from dataclasses import dataclass

    from qten.validations import need_validation, no_validate


    def check_positive(instance: "PositiveValue") -> None:
        if instance.value <= 0:
            raise ValueError("value must be positive")


    @need_validation(check_positive)
    @dataclass(frozen=True)
    class PositiveValue:
        value: int


    with no_validate():
        instance = PositiveValue(0)

    @no_validate()
    def build_intermediate():
        return PositiveValue(0)
    ```
    """

    def __enter__(self) -> "no_validate":
        previous_depth = _validation_disabled_depth()
        _VALIDATION_STATE.disabled_depth = previous_depth + 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        previous_depth = _validation_disabled_depth() - 1
        if previous_depth == 0:
            delattr(_VALIDATION_STATE, "disabled_depth")
        else:
            _VALIDATION_STATE.disabled_depth = previous_depth


def validate(v: Any) -> None:
    """
    Run the configured validator chain for ``v`` immediately.

    Unlike construction-time validation, this function ignores
    [`no_validate`][qten.validations.no_validate] and always executes the
    validators attached to `type(v)`. Validators run in the same order used by
    construction-time validation: inherited validators first, followed by
    validators declared on the concrete class.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the type of ``v`` is not configured for validation.

    Examples
    --------
    ```python
    from dataclasses import dataclass

    from qten.validations import need_validation, validate


    def check_nonempty(instance: "Label") -> None:
        if not instance.name:
            raise ValueError("name must be non-empty")


    @need_validation(check_nonempty)
    @dataclass(frozen=True)
    class Label:
        name: str


    label = Label("site")
    validate(label)
    ```
    """
    cls = type(v)
    if not hasattr(cls, "__validators__"):
        raise TypeError(
            f"{cls.__name__} is not configured for validation. "
            "Decorate the dataclass with @need_validation(...) before calling validate()."
        )
    for validator in _class_validators(cls):
        validator(v)


def need_validation(
    *validators: ValidatorFn,
) -> Callable[[_C], _C]:
    """
    Decorate a dataclass so its ``__init__`` runs inherited validators.

    The decorator is dataclass-only. It materializes ``__validators__`` on the
    decorated class by collecting validator lists from the MRO in base-to-
    derived order, appending any validators passed directly to the decorator,
    and deduplicating by validator object identity.

    Validation happens after the original dataclass `__init__` completes, which
    means dataclass `__post_init__` normalization has already run. Use
    [`no_validate`][qten.validations.no_validate] to suppress only automatic
    construction-time validation; explicit calls to
    [`validate()`][qten.validations.validate] still run validators.

    Validator contract
    ------------------
    A validator is any callable with the signature `validator(instance) -> None`.
    It should return `None` when the instance is valid and raise a user-facing
    exception, usually `TypeError` or `ValueError`, when validation fails.

    Parameters
    ----------
    *validators : ValidatorFn
        Validator callables to append to the validators inherited from base
        classes. Passing no validators is valid and preserves inherited
        validators on the decorated class.

    Returns
    -------
    Callable[[_C], _C]
        A class decorator that attaches the combined validator chain to the
        dataclass and returns the same class object.

    Raises
    ------
    TypeError
        If the decorator is applied to a class that is not a dataclass.

    Examples
    --------
    ```python
    from dataclasses import dataclass

    from qten.validations import need_validation


    def check_positive(instance: "PositiveValue") -> None:
        if instance.value <= 0:
            raise ValueError("value must be positive")


    @need_validation(check_positive)
    @dataclass(frozen=True)
    class PositiveValue:
        value: int


    PositiveValue(1)
    ```
    """

    def decorator(cls: _C) -> _C:
        if not is_dataclass(cls):
            raise TypeError("@need_validation can only be applied to dataclasses.")

        inherited: List[ValidatorFn] = []
        for base in reversed(cls.__mro__[1:]):
            base_validators = cast(
                tuple[ValidatorFn, ...], getattr(base, "__validators__", ())
            )
            for validator in base_validators:
                if validator not in inherited:
                    inherited.append(validator)

        combined = inherited.copy()
        for validator in validators:
            if validator not in combined:
                combined.append(validator)
        setattr(cls, "__validators__", combined)

        original_init = cls.__init__

        def run_validators(self: Any) -> None:
            if _validation_disabled_depth() > 0:
                return
            for validator in _class_validators(cls):
                validator(self)

        @wraps(original_init)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            run_validators(self)

        setattr(cls, "__init__", wrapped)
        return cls

    return decorator
