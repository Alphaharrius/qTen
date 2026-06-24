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
