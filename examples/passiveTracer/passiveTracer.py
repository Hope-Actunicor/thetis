"""
TELEMAC-2D point discharge with diffusion test case
===================================================

Solves tracer advection equation in a rectangular domain with
uniform fluid velocity, constant diffusivity and a constant
tracer source term. Neumann conditions are imposed on the
channel walls and a Dirichlet condition is imposed on the
inflow boundary, with the outflow boundary remaining open.

The two different functional quantities of interest considered
in [2] are evaluated on each mesh and convergence is assessed.

Further details can be found in [1].

[1] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system:
    2D hydrodynamics TELEMAC-2D software release 7.0 user
    manual." Paris:  R&D, Electricite de France, p. 134
    (2014).

[2] J.G. Wallwork, N. Barral, D.A. Ham, M.D. Piggott,
    "Anisotropic Goal-Oriented Mesh Adaptation in Firedrake",
    In: Proceedings of the 28th International Meshing
    Roundtable (2020), DOI:10.5281/zenodo.3653101,
    https://doi.org/10.5281/zenodo.3653101.
"""
from thetis import *
import numpy as np


def bessi0(x):
    """Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'."""
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = (exp(ax)/sqrt(ax))*(0.39894228 + y2*(0.1328592e-1 + y2*(0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(-0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)


def bessk0(x):
    """Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'."""
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)


class PassiveTracerParameters():
    def __init__(self, setup=1):
        assert setup in (1, 2)
        self.setup = setup

        # Parametrisation of point source
        # NOTE: Delta functions are not in H1 and hence do not live in the FunctionSpaces we seek
        #       to use. The idea here is to approximate the delta function using a disc with a very
        #       small radius. This works well for practical applications, but is not quite right for
        #       analytical test cases such as this. As such, we have calibrated the disc radius so
        #       that solving on a sequence of increasingly refined uniform meshes leads to
        #       convergence of the uniform mesh solution to the analytical solution.
        self.source_x, self.source_y = 2.0, 5.0
        self.source_r = 0.07980 if setup == 1 else 0.07972
        # TODO: Calibrate for this finite element space
        self.source_value = 100.0

        # Physical parameters
        self.diffusivity = Constant(0.1)
        self.viscosity = Constant(1.0e-08)
        self.drag = Constant(0.0025)
        self.uv = Constant(as_vector([1.0, 0.0]))
        self.elev = Constant(0.0)
        self.bathymetry = Constant(5.0)

        # Boundary conditions
        self.boundary_conditions = {
            'tracer': {
                1: {'value': Constant(0.0)},      # inflow
                2: {'open': None},                # outflow
                3: {'diff_flux': Constant(0.0)},  # Neumann
                4: {'diff_flux': Constant(0.0)},  # Neumann
            },
            'shallow_water': {
                1: {'uv': Constant(as_vector([1.0, 0.0])), 'elev': Constant(0.0)},  # inflow
                2: {'uv': Constant(as_vector([1.0, 0.0])), 'elev': Constant(0.0)},  # outflow
                3: {'un': Constant(0.0)},                    # free-slip
                4: {'un': Constant(0.0)},                    # free-slip
            }
        }

    def ball(self, mesh, triple, scaling=1.0, eps=1.0e-10):
        x, y = SpatialCoordinate(mesh)
        expr = lt((x-triple[0])**2 + (y-triple[1])**2, triple[2]**2 + eps)
        return conditional(expr, scaling, 0.0)

    def source(self, fs):
        triple = (self.source_x, self.source_y, self.source_r)
        area = assemble(self.ball(fs.mesh(), triple)*dx)
        area_exact = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        scaling *= 0.5*self.source_value
        return self.ball(fs.mesh(), triple, scaling=scaling)

    def quantity_of_interest_kernel(self, mesh):
        triple = (20.0, 5.0 if self.setup == 1 else 7.5, 0.5)
        area = assemble(self.ball(mesh, triple)*dx)
        area_exact = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        return self.ball(mesh, triple, scaling=scaling)

    def quantity_of_interest(self, sol):
        kernel = self.quantity_of_interest_kernel(sol.function_space().mesh())
        return assemble(inner(kernel, sol)*dx(degree=12))

    def exact_quantity_of_interest(self, mesh):
        x, y = SpatialCoordinate(mesh)
        x0, y0, r = self.source_x, self.source_y, self.source_r
        nu = self.diffusivity
        # q = 0.01  # sediment discharge of source (kg/s)
        q = 1
        r = max_value(sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)), r)  # (Bessel fn explodes at (x0, y0))
        sol = 0.5*q/(pi*nu)*exp(0.5*self.uv[0]*(x-x0)/nu)*bessk0(0.5*self.uv[0]*r/nu)
        kernel = self.quantity_of_interest_kernel(mesh)
        return assemble(kernel*sol*dx(degree=12))


def solve_tracer(n, setup=1, hydrodynamics=False):
    mesh2d = RectangleMesh(100*2**n, 20*2**n, 50, 10)
    P1_2d = FunctionSpace(mesh2d, "CG", 1)

    # Set up parameter class
    params = PassiveTracerParameters(setup=setup)
    source = params.source(P1_2d)

    # Solve tracer transport problem
    solver_obj = solver2d.FlowSolver2d(mesh2d, params.bathymetry)
    options = solver_obj.options
    options.timestepper_type = 'SteadyState'
    options.timestep = 20.0
    options.simulation_end_time = 18.0
    options.simulation_export_time = 18.0
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = None
    options.fields_to_export = ['tracer_2d', 'uv_2d', 'elev_2d']

    # Hydrodynamics
    options.element_family = 'dg-dg'
    options.horizontal_diffusivity = params.diffusivity
    options.horizontal_viscosity = params.viscosity
    options.quadratic_drag_coefficient = params.drag
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)

    # Passive tracer
    options.solve_tracer = True
    options.tracer_only = not hydrodynamics
    options.use_lax_friedrichs_tracer = True
    options.lax_friedrichs_tracer_scaling_factor = Constant(1.0)
    options.tracer_source_2d = source

    # Initial and boundary conditions
    solver_obj.bnd_functions = params.boundary_conditions
    uv_init = Constant(as_vector([1.0e-08, 0.0])) if hydrodynamics else params.uv
    solver_obj.assign_initial_conditions(tracer=source, uv=uv_init, elev=params.elev)
    solver_obj.iterate()

    # Evaluate quantity of interest
    sol = solver_obj.fields.tracer_2d
    print_output("J{:d}      : {:.4e}".format(setup, params.quantity_of_interest(sol)))
    print_output("J{:d}_exact: {:.4e}".format(setup, params.exact_quantity_of_interest(mesh2d)))


if __name__ == "__main__":
    refinement_level = 1
    hydrodynamics = True
    setup = 1
    solve_tracer(refinement_level, setup=setup, hydrodynamics=hydrodynamics)
