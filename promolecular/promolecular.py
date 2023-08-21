import numpy as np
from scipy.special import erf
from importlib_resources import files


__all__ = ["Promolecular"]

class Promolecular:
    def __init__(self, atom_nums, mol_coords, anion=False, cation=False):
        if anion and cation:
            raise ValueError(f"Both anion and cation can't be true.")
        self.atom_nums = atom_nums
        self.mol_coords = mol_coords
        self.anion = anion
        self.cation = cation
        # These have shape (M, K_i) where M is the number of atoms.
        self.coeffs_s, self.coeffs_p, self.exps_s, self.exps_p = self.load_coefficients_exponents()

    @property
    def numb_atoms(self):
        return len(self.mol_coords)

    def load_coefficients_exponents(self):
        if self.anion:
            path = files("promolecular.data").joinpath("kl_slsqp_results_anion.npz")
            all_results = np.load(path)
        elif self.cation:
            path = files("promolecular.data").joinpath("kl_slsqp_results.npz")
            all_results_neutral = np.load(path)
            path = files("promolecular.data").joinpath("kl_slsqp_results_cation.npz")
            all_results = np.load(path)
        else:
            path = files("promolecular.data").joinpath("kl_slsqp_results.npz")
            all_results = np.load(path)
        coeffs_s, coeffs_p = [], []
        exps_s, exps_p = [], []

        atoms = np.array(["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
                 "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca",
                 "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu",
                 "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr",
                 "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag",
                 "cd", "in", "sn", "sb", "te", "i", "xe"])
        atomic_numbs = np.array([1 + i for i in range(0, len(atoms))])
        atoms_id = [atoms[np.where(atomic_numbs == int(x))[0]][0] for x in self.atom_nums]
        for atom in atoms_id:
            if atom == "h" and self.cation:
                coeffs = all_results_neutral[atom + "_coeffs"]
                exps = all_results_neutral[atom + "_exps"]
                num_s = all_results_neutral[atom + "_num_s"]
            else:
                coeffs = all_results[atom + "_coeffs"]
                exps = all_results[atom + "_exps"]
                num_s = all_results[atom + "_num_s"]

            coeffs_s.append(coeffs[:num_s])
            coeffs_p.append(coeffs[num_s:])

            exps_s.append(exps[:num_s])
            exps_p.append(exps[num_s:])

        return np.array(coeffs_s), np.array(coeffs_p), np.array(exps_s), np.array(exps_p)

    def compute_density(self, pts):
        r"""


        Notes
        -----
        Approach One - For-loop over atom


        """
        density = np.zeros(len(pts), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # Center the points to the atom
            centered_pts = np.sum((pts - center)**2.0, axis=1)
            # Has shape (K, N), where K number of exponentials and N number of points
            exponential = np.exp(-exps[:, None] * centered_pts)
            normalization = (exps / np.pi)**1.5
            # Calculate S-type Gaussians
            density += np.einsum(
                "i,i,ij->j", coeffs, normalization, exponential, optimize=True
            )

            # Calculate P-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            if len(coeffs) != 0:
                exponential = np.exp(-exps[:, None] * centered_pts)
                normalization = (2.0 * exps**2.5) / (3.0 * np.pi**1.5)
                density += np.einsum(
                    "i,i,j,ij->j",
                    coeffs, normalization, centered_pts, exponential, optimize=True
                )
        return density

    def compute_gradient(self, pts):
        gradient = np.zeros((len(pts), 3), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # Center the points to the atom
            diff = pts - center
            r_2 = np.sum(diff ** 2.0, axis=1)
            # Has shape (K, N), where K number of exponentials and N number of points
            exponential = np.exp(-exps[:, None] * r_2)
            normalization = (exps / np.pi) ** 1.5
            # Gradient pre-fact (-2.0 * exp * (x - C))
            prefact = -2.0 * np.einsum("i,jk->ijk", exps, diff)
            gradient += np.einsum(
                "i,i,ijk,ij->jk", coeffs, normalization, prefact, exponential, optimize=True
            )

            # Calculate P-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            if len(coeffs) != 0:
                exponential = np.exp(-exps[:, None] * r_2)
                normalization = (2.0 * exps ** 2.5) / (3.0 * np.pi ** 1.5)
                # Gradient pre-fact (-2.0 * exp * x) * x^2 + 2.0 * x
                prefact = np.einsum("j,i,jk->ijk", r_2, 2.0 * exps, diff, optimize=True)
                prefact = 2.0 * diff - prefact
                gradient += np.einsum(
                    "i,i,ijk,ij->jk",
                    coeffs, normalization, prefact, exponential, optimize=True
                )

        return gradient

    def compute_laplacian(self, pts):
        lap = np.zeros(len(pts), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]

            # Calculate Laplacian of promolecular of s-type Gaussians
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # Center the points to the atom
            diff = pts - center
            r_2 = np.sum(diff ** 2.0, axis=1)
            # Has shape (K, N), where K number of exponentials and N number of points
            exponential = np.exp(-exps[:, None] * r_2)
            normalization = (exps / np.pi) ** 1.5
            prefact = 4.0 * np.einsum("i,jk->ijk", exps, diff**2.0) - 2.0
            lap += np.einsum(
                "i,i,i,ijk,ij->j", coeffs, normalization, exps, prefact, exponential, optimize=True
            )

            # Calculate Laplacian of promolecular of p-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            if len(coeffs) != 0:
                exponential = np.exp(-exps[:, None] * r_2)
                normalization = (2.0 * exps ** 2.5) / (3.0 * np.pi ** 1.5)
                common_factor = 1.0 - exps[:, None] * r_2
                diff_sq = diff**2.0
                prefact = -4.0 * np.einsum("i,jk,ij->ijk", exps, diff_sq, common_factor)
                prefact += 2.0 * common_factor[:, :, None]
                prefact -= 4.0 * diff_sq[None, :, :] * exps[:, None, None]
                lap += np.einsum(
                    "i,i,ijk,ij->j", coeffs, normalization, prefact, exponential, optimize=True
                )

        return lap

    def compute_esp(self, pts):
        esp = np.zeros(len(pts), dtype=pts.dtype)
        for i_atom in range(self.numb_atoms):
            center = self.mol_coords[i_atom]

            # Calculate ESP of promolecular of s-type Gaussians
            coeffs = self.coeffs_s[i_atom]
            exps = self.exps_s[i_atom]
            # let p = (a/pi)^1.5 e^(-a r_C^2)
            #  then the esp integral: \int p(r) / |r - P|^2 = erf(sqrt(a) R_{PC}) / R_{PC}
            r_pc = np.linalg.norm(pts - center, axis=1)
            erf_func = erf(np.sqrt(exps[:, None]) * r_pc)
            esp += np.einsum("i,ij,j->j", coeffs, erf_func, 1.0 / r_pc, optimize=True)

            # Calculate ESP of promolecular of p-type Gaussians
            coeffs = self.coeffs_p[i_atom]
            exps = self.exps_p[i_atom]
            if len(coeffs) != 0:
                etf_func = erf(np.sqrt(exps[:, None]) * r_pc) / r_pc
                exponential = np.exp(-exps[:, None] * r_pc)
                normalization = (2.0 * exps ** 2.5) / (3.0 * np.pi ** 1.5)
                # \int r^2 e^{-a r^2} / |r - P| = N^{-1} etf(sqrt(a) R_{PC}) / R_{PC} - pi e^{-a R_{PC}^2} / a^2
                #   where N is the normalization constant above
                integral = etf_func - normalization[:, None] * np.pi * exponential / exps[:, None]**2.0
                esp += np.einsum("i,ij->j", coeffs, integral, optimize=True)
        print(esp)
        centered = np.linalg.norm(self.mol_coords[:, None, :] - pts[None, :, :], axis=2)
        # i is number of atoms,    j is the points,
        return np.einsum("i,ij->j", self.atom_nums, 1.0 / centered) - esp
