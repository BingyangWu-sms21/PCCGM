r"""
FEniCS-based differentiable solver for 2D plain strain Periodic Boundary
Condition (PBC) problem in a Representative Volume Element (RVE) with a
given global strain.
"""
from fenics import *
from fenics_adjoint import *
import torch
import torch_fenics
from typing import Tuple, List, Optional
import numpy as np


class PeriodicBoundary2D(SubDomain):
    r"""
    2D periodic boundary condition in both directions. The left-bottom corner of
    the domain is (0, 0).

    Args:
        corner (Tuple[float, float]): The right-top corner of the domain.
        tolerance (float): Tolerance for detecting the boundary.
    """
    def __init__(self, corner: Tuple[float, float], tolerance: float = DOLFIN_EPS):
        super().__init__(tolerance)
        self.tol = tolerance
        self.corner = corner

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary and not on the top-left corner
        # or right-bottom corner
        return bool((near(x[0], 0.0, self.tol) and not near(x[1], self.corner[1], self.tol)) or
                    (near(x[1], 0.0, self.tol) and not near(x[0], self.corner[0], self.tol)) and
                    on_boundary)

    def map(self, x, y):
        if near(x[0], self.corner[0], self.tol):
            y[0] = x[0] - self.corner[0]
        else:
            y[0] = x[0]
        if near(x[1], self.corner[1], self.tol):
            y[1] = x[1] - self.corner[1]
        else:
            y[1] = x[1]


class PeriodicBCRVE2D(torch_fenics.FEniCSModule):
    r"""
    Class for solving 2D plain strain Periodic Boundary Condition (PBC) problem
    in a Representative Volume Element (RVE) with a given global strain list.
    Written based on https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html

    Args:
        corner (Tuple[float, float]): The right-top corner of the domain.
        n_cells (Tuple[int, int]): The number of cells in x and y directions.
        global_strain_list (List[Tuple[float, float, float]]): The list of global
            strains, in the form of Voigt representation: (\epsilon_{xx},
            \epsilon_{yy}, \gamma_{xy} = 2 \epsilon_{xy}).
    """
    def __init__(self,
                 corner: Tuple[float, float],
                 n_cells: Tuple[int, int],
                 global_strain_list: List[Tuple[float, float, float]],
                 sol_vec_elem: Optional[VectorElement] = None,
                 ):
        super().__init__()

        # Create function space
        if len(corner) != 2 or len(n_cells) != 2:
            raise ValueError("corner and n_cells must have length 2")
        self.corner = corner
        self.vol = corner[0] * corner[1]  # volume (area) of the domain
        self.n_cells = n_cells
        self.mesh = RectangleMesh(Point(0.0, 0.0), Point(*corner), *n_cells)
        if sol_vec_elem is None:
            Ve = VectorElement("Lagrange", self.mesh.ufl_cell(), 1)
        else:
            Ve = sol_vec_elem

        Re = VectorElement("R", self.mesh.ufl_cell(), 0)

        Fe = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)

        self.V = FunctionSpace(self.mesh, MixedElement([Ve, Re]),
                               constrained_domain=
                               PeriodicBoundary2D(corner, 1e-10))
        self.W = FunctionSpace(self.mesh, Fe,
                               constrained_domain=
                               PeriodicBoundary2D(corner, 1e-10))  # for material properties
        self.S = FunctionSpace(self.mesh, Re, constrained_domain=
                               PeriodicBoundary2D(corner, 1e-10))
        self.U = FunctionSpace(self.mesh, Ve, constrained_domain=
                               PeriodicBoundary2D(corner, 1e-10))

        # map index of spatial coordinates to dof index of W
        prop_dof_coordinates = self.W.tabulate_dof_coordinates()
        # dof index to spatial coordinate index
        prop_sorted_index = np.lexsort((prop_dof_coordinates[:, 1], prop_dof_coordinates[:, 0]))
        # get the inverse mapping
        prop_inverse_index = np.argsort(prop_sorted_index)
        self.prop_inverse_index = torch.tensor(prop_inverse_index)

        # construct a function to identify component of dof (x or y or scalar)
        vec_comp = Function(self.U)
        vec_comp.interpolate(Constant((-1.0, 1.0)))
        s_comp = Function(self.S)
        s_comp.interpolate(Constant((0.0, 0.0)))
        assigner = FunctionAssigner(self.V, [self.U, self.S])
        comp_func = Function(self.V)
        assigner.assign(comp_func, [vec_comp, s_comp])
        comp_vec = comp_func.vector().get_local()
        x_index = np.where(np.isclose(comp_vec, -1.0))[0]
        y_index = np.where(np.isclose(comp_vec, 1.0))[0]

        # reorder dofs to spatial coordinate order
        dof_coordinates = self.V.tabulate_dof_coordinates()
        x_dof_coordinates = dof_coordinates[x_index]
        y_dof_coordinates = dof_coordinates[y_index]

        x_sorted_index = np.lexsort((x_dof_coordinates[:, 1], x_dof_coordinates[:, 0]))

        # only save regular grid points
        x_filtered_sorted_index = []
        for idx in x_sorted_index:
            x = x_dof_coordinates[idx, 0]
            y = x_dof_coordinates[idx, 1]
            x_step = corner[0] / n_cells[0]
            y_step = corner[1] / n_cells[1]
            if near(x / x_step, round(x / x_step)) and near(y / y_step, round(y / y_step)):
                    x_filtered_sorted_index.append(idx)

        self.x_sorted_index = torch.tensor(x_index[x_filtered_sorted_index])
        if len(self.x_sorted_index) != n_cells[0] * n_cells[1]:
            print(len(self.x_sorted_index))
            raise RuntimeError("Number of regular grid points does not match")

        y_sorted_index = np.lexsort((y_dof_coordinates[:, 1], y_dof_coordinates[:, 0]))

        # only save regular grid points
        y_filtered_sorted_index = []
        for idx in y_sorted_index:
            x = y_dof_coordinates[idx, 0]
            y = y_dof_coordinates[idx, 1]
            x_step = corner[0] / n_cells[0]
            y_step = corner[1] / n_cells[1]
            if near(x / x_step, round(x / x_step)) and near(y / y_step, round(y / y_step)):
                    y_filtered_sorted_index.append(idx)

        self.y_sorted_index = torch.tensor(y_index[y_filtered_sorted_index])
        if len(self.y_sorted_index) != n_cells[0] * n_cells[1]:
            print(len(self.y_sorted_index))
            raise RuntimeError("Number of regular grid points does not match")

        # merge x_sorted_index and y_sorted_index
        self.sorted_index = torch.stack(
            (self.x_sorted_index, self.y_sorted_index), dim=1).reshape(-1)

        # check global strain list
        for global_strain in global_strain_list:
            if len(global_strain) != 3:
                raise ValueError("global strain must have length 3")
        self.global_strain_list = global_strain_list

    def solve(self, lmbda, mu) -> Tuple:

        # construct bilinear form
        def epsilon(u):
            return sym(grad(u))

        def sigma(u, Eps):
            eps = epsilon(u) + Eps  # add global strain
            return lmbda * tr(eps) * Identity(2) + 2.0 * mu * eps

        def assign_global_strain(Eps, global_strain):
            Eps.assign(Constant(((global_strain[0], global_strain[2] / 2.0),
                                (global_strain[2] / 2.0, global_strain[1]))))

        u, alpha = TrialFunctions(self.V)
        v, beta = TestFunctions(self.V)
        Eps = Constant(((1.0, 0.0), (0.0, 0.0)))
        F = inner(sigma(u, Eps), epsilon(v)) * dx
        a, L = lhs(F), rhs(F)
        a += dot(alpha, v) * dx + dot(beta, u) * dx

        u_sol_list = []
        avg_stress_list = []
        for global_strain in self.global_strain_list:
            assign_global_strain(Eps, global_strain)
            w_sol = Function(self.V)
            solve(a == L, w_sol, [], solver_parameters={"linear_solver": "cg"})
            u_sol, _ = w_sol.split()
            u_sol_list.append(u_sol)
            sigma_sol = sigma(u_sol, Eps)
            avg_stress = [assemble(sigma_sol[0, 0] * dx) / self.vol,
                          assemble(sigma_sol[1, 1] * dx) / self.vol,
                          assemble(sigma_sol[0, 1] * dx) / self.vol]
            avg_stress_list.extend(avg_stress)

        return tuple(u_sol_list) + tuple(avg_stress_list)

    def run_and_process(self,
                        lmbda_tensor: torch.Tensor,
                        mu_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Run the solver with the given material properties and process the results.

        Args:
            lmbda_tensor (torch.Tensor): The tensor of Lame's first parameter. Shape
                (n_batch, n_cells[0], n_cells[1]).
            mu_tensor (torch.Tensor): The tensor of shear modulus. Shape
                (n_batch, n_cells[0], n_cells[1]).

        Returns:
            u_sol_tensor (torch.Tensor): The tensor of displacement fields. Shape
                (n_batch, n_global_strains, n_cells[0], n_cells[1], 2).
            avg_stress_tensor (torch.Tensor): The tensor of average stress fields.
                Shape (n_batch, n_global_strains, 3).
        """
        n_cases = len(self.global_strain_list)
        if lmbda_tensor.dim() != 3 or mu_tensor.dim() != 3:
            raise ValueError("lmbda_tensor and mu_tensor must have shape (n_batch, n_cells[0], n_cells[1])")
        if lmbda_tensor.shape != mu_tensor.shape:
            raise ValueError("lmbda_tensor and mu_tensor must have the same shape")
        n_batch = lmbda_tensor.shape[0]
        lmbda_tensor = lmbda_tensor.reshape(n_batch, -1)
        lmbda_tensor = lmbda_tensor[:, self.prop_inverse_index]
        mu_tensor = mu_tensor.reshape(n_batch, -1)
        mu_tensor = mu_tensor[:, self.prop_inverse_index]
        results = self(lmbda_tensor, mu_tensor)
        u_sol_list = results[:n_cases]
        avg_stress_list = results[n_cases:]
        # Shape: n_cases number of (n_batch, n_dofs, 2) -> (n_batch, n_cases, n_x, n_y, 2)
        u_sol_tensor = torch.stack([self.get_2d_tensor(u_sol) for u_sol in u_sol_list], dim=1)
        # Shape: n_cases * 3 number of (n_batch) -> (n_batch, n_cases, 3)
        avg_stress_tensor = torch.stack(avg_stress_list, dim=1).reshape(-1, n_cases, 3)
        return u_sol_tensor, avg_stress_tensor

    def input_templates(self):
        return Function(self.W), Function(self.W)

    def get_2d_tensor(self, f: torch.Tensor) -> torch.Tensor:
        r"""
        Convert a plain tensor obtained from Function.vector().get_local() to a
        tensor with the correct spatial structure.
        If high-order elements are used, number of dofs may exceed the number of
        regular grid points. In this case, only the dofs corresponding to the
        regular grid points are returned.

        Args:
            f (torch.Tensor): The plain tensor. Shape (n_batch, n_dofs,) or
            (n_batch, n_dofs // n_components, n_components).

        Returns:
            The tensor with the correct spatial structure. Shape (n_batch,
            n_x, n_y) or (n_batch, n_x, n_y, n_components),
            respectively.
        """
        if f.dim() == 2:
            target_shape = (f.shape[0], *self.n_cells)
        elif f.dim() == 3:
            target_shape = (f.shape[0], *self.n_cells, f.shape[2])
            f = f.reshape(f.shape[0], -1)
        else:
            raise ValueError("f must have shape (n_batch, n_dofs,) or "
                             "(n_batch, n_dofs, n_components)")

        return f[:, self.sorted_index].reshape(target_shape)


def main():
    import matplotlib.pyplot as plt
    def in_inclusion(x, y, center, radius):
        return (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2
    n_x = 128
    n_y = 128
    corner = (1.0, 1.0)
    n_cells = (n_x, n_y)
    center_inclusion = (0.5, 0.5)
    radius_inclusion = 0.2
    global_strain_list = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    solver = PeriodicBCRVE2D(
        corner=corner, n_cells=n_cells, global_strain_list=global_strain_list)
    E_matrix = 71.0
    nu_matrix = 0.33
    E_inclusion = 400.0
    nu_inclusion = 0.25
    lmbda_matrix = E_matrix * nu_matrix / ((1.0 + nu_matrix) * (1.0 - 2.0 * nu_matrix))
    mu_matrix = E_matrix / (2.0 * (1.0 + nu_matrix))
    lmbda_inclusion = E_inclusion * nu_inclusion / ((1.0 + nu_inclusion) * (1.0 - 2.0 * nu_inclusion))
    mu_inclusion = E_inclusion / (2.0 * (1.0 + nu_inclusion))
    # construct the label field, 1 for inclusion, 0 for matrix
    x = np.linspace(0.0, corner[0], n_x + 1)[:-1]
    y = np.linspace(0.0, corner[1], n_y + 1)[:-1]
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    label_field = in_inclusion(x_grid, y_grid, center_inclusion, radius_inclusion)
    # plot the label field
    plt.imshow(label_field, origin="lower")
    plt.savefig("label_field.png")
    label_field_tensor = torch.tensor(label_field, dtype=torch.float64, requires_grad=True).unsqueeze(0)
    lmbda = lmbda_matrix * (1.0 - label_field_tensor) + lmbda_inclusion * label_field_tensor
    mu = mu_matrix * (1.0 - label_field_tensor) + mu_inclusion * label_field_tensor
    u_sol, avg_stress = solver.run_and_process(lmbda, mu)
    print(avg_stress)


if __name__ == "__main__":
    main()
