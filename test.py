from pyhilbert.spatials import Lattice
from sympy import ImmutableDenseMatrix
from pyhilbert.basis_transform import BasisTransform

basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
lattice = Lattice(basis=basis, shape=(2, 2), unit_cell={"r": ImmutableDenseMatrix([[0], [0]])})
t = BasisTransform(ImmutableDenseMatrix([[1, 1], [0, 1]]))
lattice.plot("structure",backend="plotly",show=True)
new_lattice = t(lattice)
new_lattice.plot("structure",backend="plotly",show=True)