"""
Validation decorators and symbolic validation helpers.

This package centralizes the runtime validation layer used across QTen
dataclasses and symbolic objects. The core API lets a dataclass declare
validators once, run them automatically after construction, run them manually
on an existing object, or temporarily suppress construction-time validation
when building intermediate states.

Core exports
------------
- [`need_validation()`][qten.validations.need_validation]
  Attach validators to a dataclass and run them after `__init__`.
- [`validate()`][qten.validations.validate]
  Execute registered validators explicitly on an existing instance.
- [`no_validate`][qten.validations.no_validate]
  Temporarily suppress construction-time validation in the current thread.

Related module
--------------
- [`qten.validations.symbolics`][qten.validations.symbolics]
  Symbolics-specific validator factories used by geometry and Hilbert-space
  objects.
"""

from ._validations import (
    need_validation as need_validation,
    no_validate as no_validate,
    validate as validate,
)
