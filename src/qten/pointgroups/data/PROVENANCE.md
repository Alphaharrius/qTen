# Point-Group Data Provenance

The normalized point-group generator data in `point_group_data.json`
was generated from pymatgen-core `v2026.5.18`:

- https://raw.githubusercontent.com/materialsproject/pymatgen-core/v2026.5.18/src/pymatgen/symmetry/symm_data.json
- https://cryst.ehu.es/cgi-bin/rep/programs/sam/point.py?num=<point-group-number>&sg=<representative-space-group>

The runtime JSON is normalized into one qten-owned record per
crystallographic point group. Each record contains:

- `symbol`
- `aliases`
- `crystal_system`
- `source_encoding`
- `generators`
- `irreps` from Bilbao character tables

Schoenflies aliases are added by qten for convenient lookup.
Preserve pymatgen attribution and license terms when redistributing this data.

## Spinor irreducible representations

`double_point_group_data.json` is generated offline from the packaged
crystallographic generators by `scripts/retrieve_double_pointgroup_data.py`
using spgrep 0.6.0:

- https://github.com/spglib/spgrep

spgrep enumerates the projective spinor irreducible representations and their
factor systems from each standard crystallographic lattice and spatial group.
The generator converts spgrep's operation-wise SU(2) section to QTen's
`qten-su2-principal-v1` convention and stores per-operation characters. No
Bilbao double-group tables or labels are copied into this cache.

spgrep is a build-time development dependency only. Runtime table lookup does
not import spgrep. The cache records the spgrep version, source-data SHA-256,
generator SHA-256 values, factor systems, and character rounding precision so
regeneration can be checked deterministically.

@article{spgrep,
    doi = {10.21105/joss.05269},
    url = {https://doi.org/10.21105/joss.05269},
    year = {2023},
    publisher = {The Open Journal},
    volume = {8},
    number = {85},
    pages = {5269},
    author = {Shinohara, Kohei and Togo, Atsushi and Tanaka, Isao},
    title = {spgrep: On-the-fly generator of space-group irreducible representations},
    journal = {J. Open Source Softw.}
}