# Point-Group Data Provenance

The catalog in `point_group_data.json` has one record shape for
ordinary and spinor characters: generators plus class-wise χ.

## Ordinary irreducible representations

3D generators come from pymatgen-core `v2026.5.18`:

- https://raw.githubusercontent.com/materialsproject/pymatgen-core/v2026.5.18/src/pymatgen/symmetry/symm_data.json

Ordinary `irreps` are a pinned Bilbao class-table cache:

- https://cryst.ehu.es/cgi-bin/rep/programs/sam/point.py?num=<point-group-number>&sg=<representative-space-group>

Linear characters of the 32 crystallographic point groups do not
depend on QTen's SU(2) lift, so these names and numbers stay frozen.
A catalog rebuild only recomputes `spinor_irreps`. `--check` confirms
that the generated group still realizes the packaged ordinary table.
Schoenflies aliases are added by qten.

## Spinor irreducible representations

`spinor_irreps` uses the same `class_labels` / `{name: {dim, characters}}`
shape. The numbers are computed from QTen's principal SU(2) lift
`qten-su2-principal-v1` of each element's `rotation3`. They are not
copied from Bilbao double-group pages (those tables have no U(g), so a
section cannot be aligned).

2D and 1D records add `dim`, `frame`, and paired `spatial` / `rotation3`
generators. Spin still uses the stored 3D rotation, never a padded 2D
matrix.

## spgrep

spgrep is a development-only checker (`--check-spgrep`): after η-aligning
its SU(2) section to QTen's lift, class averages are compared. spgrep
numbers are never written into this JSON.

- https://github.com/spglib/spgrep

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
