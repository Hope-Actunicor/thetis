"""
Steady-state channel flow in 3D
===============================

Steady state flow in a channel subject to bottom friction.

This test reproduces the "channel" test case found in GOTM test suite [1]
and also [2].

This case tests the turbulence closure model, vertical viscosity and bottom
boundary layer. Water column is initially at rest. Circulation is driven by
a constant elevation gradient until it reaches a steady state. Here the
elevation gradient is replaced by an equivalent source term in the
momentum equation.


[1] http://www.gotm.net/
[2] Karna et al. (2012). Coupling of a discontinuous Galerkin finite element
    marine model with a finite difference turbulence closure model.
    Ocean Modelling, 47:55-64.
    http://dx.doi.org/10.1016/j.ocemod.2012.01.001

Tuomas Karna 2016-03-10
"""
from thetis import *
import numpy

physical_constants['z0_friction'].assign(1.5e-3)

fast_convergence = False

outputdir = 'outputs'
# set mesh resolution
dx = 2500.0
layers = 250
depth = 15.0

nx = 3  # nb elements in flow direction
lx = nx*dx
ny = 2  # nb elements in cross direction
ly = ny*dx
mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x', reorder=True)

dt = 25.0
t_end = 12 * 3600.0  # sufficient to reach ~steady state
if fast_convergence:
    t_end = 5 * 3600.0
t_export = 400.0
u_mag = 1.0

# bathymetry
p1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(p1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# create solver
solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
options = solver_obj.options
options.solve_salt = False
options.solve_temp = False
options.solve_vert_diffusion = True
options.use_bottom_friction = True
options.use_turbulence = True
options.use_parabolic_viscosity = False
options.v_viscosity = Constant(1.3e-6)  # background value
options.v_diffusivity = Constant(1.4e-7)  # background value
options.use_ale_moving_mesh = False
options.use_limiter_for_tracers = True
options.t_export = t_export
options.dt = dt
options.t_end = t_end
options.outputdir = outputdir
options.u_advection = u_mag
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'uv_dav_2d', 'uv_bottom_2d',
                            'parab_visc_3d', 'eddy_visc_3d', 'shear_freq_3d',
                            'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d', 'uv_bottom_2d',
                                 'eddy_visc_3d', 'eddy_diff_3d',
                                 'shear_freq_3d',
                                 'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
gls_options = options.gls_options
gls_options.apply_defaults('k-omega')
gls_options.stability_name = 'CB'
options.outputdir = outputdir + '_' + gls_options.closure_name + '-' + gls_options.stability_name
print_output('Exporting to ' + options.outputdir)

solver_obj.create_function_spaces()

# drive flow with momentum source term equivalent to constant surface slope
surf_slope = -1.0e-5  # d elev/dx
pressure_grad = -physical_constants['g_grav'] * surf_slope
options.uv_source_2d = Constant((pressure_grad, 0))

if fast_convergence:
    # speed-up convergence by stating with u > 0
    u_init_2d = 0.5
    solver_obj.assign_initial_conditions(uv_2d=Constant((u_init_2d, 0)))
    # consistent 3d velocity with slope
    solver_obj.fields.uv_3d.project(Expression(('u*(1.0 + 0.3*(x[2]/d + 0.5))', 0, 0), d=depth, u=u_init_2d))

if __name__ == '__main__':
    solver_obj.iterate()

    # compare against logarithmic velocity profile
    # u = u_b / kappa * log((z + bath + z_0)/z_0)
    # estimate bottom friction velocity from maximal u
    u_max = 0.9  # max velocity in [2] Fig 2.
    l2_tol = 0.05
    kappa = solver_obj.gls_model.options.kappa
    z_0 = physical_constants['z0_friction'].dat.data[0]
    u_b = u_max * kappa / np.log((depth + z_0)/z_0)
    log_uv = Function(solver_obj.function_spaces.P1DGv, name='log velocity')
    log_uv.project(Expression(('u_b / kappa * log((x[2] + depth + z_0)/z_0)', 0, 0),
                              u_b=u_b, kappa=kappa,
                              depth=depth, z_0=z_0))
    out = File(outputdir + '/log_uv.pvd')
    out.write(log_uv)

    uv_p1_dg = Function(solver_obj.function_spaces.P1DGv, name='velocity p1dg')
    uv_p1_dg.project(solver_obj.fields.uv_3d)
    volume = lx*ly*depth
    uv_l2_err = errornorm(log_uv, uv_p1_dg)/numpy.sqrt(volume)
    assert uv_l2_err < l2_tol, 'L2 error is too large: {:} > {:}'.format(uv_l2_err, l2_tol)
    print_output('L2 error {:.4f} PASSED'.format(uv_l2_err))
