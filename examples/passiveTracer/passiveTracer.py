"""
TELEMAC-2D point discharge with diffusion test case
===================================================

Solves tracer advection equation in a rectangular domain with
uniform fluid velocity, constant diffusivity and a constant
tracer source term. Neumann conditions are imposed on the
channel walls and a Dirichlet condition is imposed on the
inflow boundary, with the outflow boundary remaining open.

Further details can be found in [1].

[1] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system:
    2D hydrodynamics TELEMAC-2D software release 7.0 user
    manual." Paris:  R&D, Electricite de France, p. 134
    (2014).
"""
from thetis import *
import numpy as np


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
        self.source_value = 100.0

        # Physical parameters
        self.diffusivity = Constant(0.1)
        self.uv = Constant(as_vector([1.0, 0.0]))
        self.elev = Constant(0.0)
        self.bathymetry = Constant(1.0)

        # Boundary conditions
        neumann = {'diff_flux': Constant(0.0)}
        dirichlet = {'value': Constant(0.0)}
        outflow = {'open': None}
        self.boundary_conditions = {1: dirichlet, 2: outflow, 3: neumann, 4: neumann}

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
        source = self.ball(fs.mesh(), triple, scaling=scaling)
        return interpolate(source, fs)

    def quantity_of_interest(self, sol):
        mesh = sol.function_space().mesh()
        triple = (20.0, 5.0 if self.setup == 1 else 7.5, 0.5)
        area = assemble(self.ball(mesh, triple)*dx)
        area_exact = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        kernel = self.ball(mesh, triple, scaling=scaling)
        return assemble(inner(kernel, sol)*dx(degree=12))


def solve_tracer(n, setup=1):
    mesh2d = RectangleMesh(100*2**n, 20*2**n, 50, 10)
    P1_2d = FunctionSpace(mesh2d, "CG", 1)

    params = PassiveTracerParameters(setup=setup)
    source = params.source(P1_2d)

    solver_obj = solver2d.FlowSolver2d(mesh2d, params.bathymetry)
    options = solver_obj.options
    options.timestepper_type = 'SteadyState'
    options.timestep = 20.0
    options.simulation_end_time = 18.0
    options.simulation_export_time = 18.0
    options.fields_to_export = ['tracer_2d']
    options.solve_tracer = True
    options.use_lax_friedrichs_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = params.diffusivity
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(tracer=source, uv=params.uv)
    solver_obj.bnd_functions['tracer'] = params.boundary_conditions
    solver_obj.iterate()

    # Evaluates quantities of interest
    sol = solver_obj.fields.tracer_2d
    print_output("J{:d}: {:.4e}".format(setup, params.quantity_of_interest(sol)))


if __name__ == "__main__":
    refinement_level = 2
    for setup in (1, 2):
        solve_tracer(refinement_level, setup=setup)
