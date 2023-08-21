from promolecular.promolecular import Promolecular
from grid.onedgrid import GaussLegendre
from grid.molgrid import MolGrid
from grid.rtransform import BeckeRTransform
from grid.becke import BeckeWeights
from iodata import load_one
import pytest
import numpy as np
from scipy.optimize import approx_fprime
from scipy.stats import special_ortho_group
from importlib_resources import files

@pytest.mark.parametrize("fchk_path,cation,anion",
    [["h2o.fchk", False, True],
     ["h2o.fchk", True, False],
     ["h2o.fchk", False, False],
     ["nh3.fchk", False, False]]
)
def test_integral_of_density(fchk_path, cation, anion):
    path = files("promolecular.data.fchk").joinpath(fchk_path)
    mol = load_one(path)
    coords = mol.atcoords
    charges = mol.atnums

    promol = Promolecular(charges, coords, anion, cation)

    oned_grid = GaussLegendre(npoints=50)
    radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mol_grid = MolGrid.from_preset(
        atnums=charges,  # The atomic numbers of Formaldehyde
        atcoords=coords,  # The atomic coordinates of Formaldehyde
        rgrid=radial_grid,  # Radial grid used to construct atomic grids over each carbon, and hydrogen.
        preset="fine",
        # Controls the angular degree. Can be "coarse", "medium", "fine", "veryfine", "ultrafine", "insane".
        aim_weights=BeckeWeights(),  # Atom-in molecular weights: Becke weights
        store=True,  # Stores the individual atomic grids, used for interpolation.
    )

    density = promol.compute_density(mol_grid.points)
    integral = mol_grid.integrate(density)
    desired = np.sum(charges)
    if anion:
        desired += len(coords)
    elif cation:
        desired -= np.sum(charges != 1.0)
    assert np.abs(integral - desired) < 1e-3


@pytest.mark.parametrize("fchk_path",
    ["h2o.fchk", "nh3.fchk"]
)
def test_gradient(fchk_path):
    path = files("promolecular.data.fchk").joinpath(fchk_path)
    mol = load_one(path)
    coords = mol.atcoords
    charges = mol.atnums
    # coords = np.array([[0.0, 0.0, 0.0]])
    # charges = np.array([5])
    promol = Promolecular(charges, coords)

    oned_grid = GaussLegendre(npoints=50)
    radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mol_grid = MolGrid.from_preset(
        atnums=charges,  # The atomic numbers of Formaldehyde
        atcoords=coords,  # The atomic coordinates of Formaldehyde
        rgrid=radial_grid,  # Radial grid used to construct atomic grids over each carbon, and hydrogen.
        preset="medium",
        aim_weights=BeckeWeights(),  # Atom-in molecular weights: Becke weights
        store=True,  # Stores the individual atomic grids, used for interpolation.
    )
    true = promol.compute_gradient(mol_grid.points)
    for i, pt in enumerate(mol_grid.points):
        desired = approx_fprime(pt, lambda x: promol.compute_density(np.array([x]))[0], epsilon=1e-10)
        err = np.abs(desired - true[i])
        print(err)
        assert np.all(err < 1e-2)


@pytest.mark.parametrize("charge", [6, 7, 8])
def test_electrostatic_potential_on_atom(charge):
    r"""Test electrostatic potential on atoms is spherically symmetric and monotically decreasing."""
    # Should be constant and spherical.
    coords = np.array([[0.1, 0.0, 0.0]])
    charges = np.array([charge])
    promol = Promolecular(charges, coords)

    # Construct radial grid
    oned_grid = GaussLegendre(npoints=50)
    radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)

    # Rotate the points
    rot_mat = special_ortho_group.rvs(3)
    ray = np.vstack((radial_grid.points, np.zeros(len(radial_grid.points)), np.zeros(len(radial_grid.points)))).T
    pts_3d = coords + ray.dot(rot_mat)
    true1 = promol.compute_esp(pts_3d)
    # Make sure it is monotically decreasing where it is greater than 1e-8
    true_pos = true1[true1 >= 1e-8]
    assert np.all(true_pos[:-1] - true_pos[1:] >= 0.0)

    # Rotate the points again
    rot_mat = special_ortho_group.rvs(3)
    ray = np.vstack((radial_grid.points, np.zeros(len(radial_grid.points)), np.zeros(len(radial_grid.points)))).T
    pts_3d = coords + ray.dot(rot_mat)
    true2 = promol.compute_esp(pts_3d)

    # Make sure they are the same, i.e. spherically symmetric
    assert np.all(np.abs(true1 - true2) < 1e-8)


@pytest.mark.parametrize("fchk_path",
    ["h2o.fchk",
     "nh3.fchk",
     "atom_6",
     "atom_2"]
)
def test_laplacian(fchk_path):
    if "atom" in fchk_path:
        coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([int(fchk_path.split("_")[1])])
    else:
        path = files("promolecular.data.fchk").joinpath(fchk_path)
        mol = load_one(path)
        coords = mol.atcoords
        charges = mol.atnums

    promol = Promolecular(charges, coords)

    oned_grid = GaussLegendre(npoints=50)
    radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mol_grid = MolGrid.from_preset(
        atnums=charges,  # The atomic numbers of Formaldehyde
        atcoords=coords,  # The atomic coordinates of Formaldehyde
        rgrid=radial_grid,  # Radial grid used to construct atomic grids over each carbon, and hydrogen.
        preset="medium",
        aim_weights=BeckeWeights(),  # Atom-in molecular weights: Becke weights
        store=True,  # Stores the individual atomic grids, used for interpolation.
    )
    true = promol.compute_laplacian(mol_grid.points)
    for i, pt in enumerate(mol_grid.points):
        desired = 0.0
        for i_dim in range(3):
            f =  lambda x: promol.compute_gradient(np.array([x]))[0][i_dim]
            desired += approx_fprime(pt, f, epsilon=1e-10)[i_dim]
        print(desired)
        err = np.abs(desired - true[i])
        print("err", err)
        assert np.all(err < 1e-1)
