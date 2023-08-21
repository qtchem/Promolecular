Promolecule
===========
Promolecular model based on BFit fits of atomic densities.

```bash
pip install .
pytest -v .
```

```python
from promolecular import Promolecular

promol = Promolecular(atom_nums, mol_coords, anion=False, cation=False)
density = promol.compute_density(pts
array)
gradient = promol.compute_gradient(pts
array)
esp = promol.compute_esp(pts
array)
```