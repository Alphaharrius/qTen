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

    While this context manager is active, construction-time validators will not run.
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

    Unlike construction-time validation, this function ignores ``no_validate``
    and always executes the validators attached to ``type(v)``.

    Parameters
    ----------
    `v` : `Any`
        The value to validate.

    Raises
    ------
    `TypeError`
        If the type of ``v`` is not configured for validation.
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
