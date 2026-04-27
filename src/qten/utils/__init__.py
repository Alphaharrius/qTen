"""
General-purpose utility helpers used across QTen.

This namespace groups small infrastructural helpers that do not belong to the
main tensor, symbolic, or geometry APIs but are still useful to import from a
stable package location.

Key modules
-----------
- [`qten.utils.devices`][qten.utils.devices]
  Logical device descriptors and device-aware mixins.
- [`qten.utils.io`][qten.utils.io]
  Save/load helpers for persisted objects and experiment outputs.
- [`qten.utils.collections_ext`][qten.utils.collections_ext]
  Immutable mapping helpers such as [`FrozenDict`][qten.utils.collections_ext.FrozenDict].
- [`qten.utils.loggings`][qten.utils.loggings]
  Logging convenience functions.
- [`qten.utils.types_ext`][qten.utils.types_ext]
  Type-introspection helpers.

Notes
-----
This package currently serves mainly as a documentation and import namespace;
most concrete APIs are exposed from its child modules rather than re-exported
here.
"""
