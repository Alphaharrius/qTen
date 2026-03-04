import sympy as sy

def wrap(index: sy.Matrix, basis: sy.Matrix) -> sy.Matrix:
    """
    Wrap each component of `index` into [0, basis_i - 1] using element-wise modulo.
    """
    if index.shape != basis.shape:
        raise ValueError("Index and basis must have the same shape.")

    if any(b == 0 for b in basis):
        raise ValueError("All basis entries must be non-zero for modulo wrapping.")

    wrapped_elements = [i % b for i, b in zip(index, basis)]
    return sy.Matrix(wrapped_elements).reshape(index.rows, index.cols)


def fractional_part(vec: sy.Matrix) -> sy.Matrix:
    """Element-wise fractional part for a SymPy matrix."""
    return sy.Matrix([x - sy.floor(x) for x in vec]).reshape(vec.rows, vec.cols)


# Define an offset vector
offset = sy.Matrix([[4], [5]])
offset_fractional = fractional_part(offset)

# Define periodic lengths for each dimension
periods = sy.Matrix([[1], [1]])

# Convert offset to wrapped fractional coordinates
fractional_offset = wrap(offset_fractional, periods)

print(offset_fractional)
print(fractional_offset)
