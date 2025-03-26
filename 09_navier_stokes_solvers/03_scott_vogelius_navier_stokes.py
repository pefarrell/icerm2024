from firedrake import *

# Use a triangular mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
base = UnitSquareMesh(16, 16, diagonal="crossed", distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]
n = FacetNormal(mesh)
(x, y) = SpatialCoordinate(mesh)

# Define Scott--Vogelius function space W
V = VectorFunctionSpace(mesh, "CG", 4)
Q = FunctionSpace(mesh, "DG", 3, variant="integral")
W = MixedFunctionSpace([V, Q])

# Define Reynolds number and bcs
Re = Constant(1)
bcs = [DirichletBC(W.sub(0), Constant((0, 0)), (1, 2, 3)),
       DirichletBC(W.sub(0), as_vector([16 * x**2 * (1-x)**2, 0]), (4,))]

w = Function(W, name="Solution")
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Augmented Lagrangian term
gamma = Constant(10000)

# Define Lagrangian
L = (
      0.5 * inner(2/Re * sym(grad(u)), sym(grad(u)))*dx
    -       inner(p, div(u))*dx
    + 0.5 * gamma * inner(div(u), div(u))*dx
    )

# Optimality conditions
F = (
      derivative(L, w)
    - inner(outer(u, u), sym(grad(v)))*dx
    + inner(dot(outer(u, u), n), v)*ds
    )

sp = {
    'mat_type': 'nest',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_max_it': 20,
    'snes_atol': 1e-8,
    'snes_rtol': 1e-12,
    'snes_stol': 1e-06,
    'ksp_type': 'fgmres',
    'ksp_converged_reason': None,
    'ksp_monitor_true_residual': None,
    'ksp_max_it': 500,
    'ksp_atol': 1e-08,
    'ksp_rtol': 1e-10,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'fieldsplit_0': {'ksp_convergence_test': 'skip',
                     'ksp_max_it': 1,
                     'ksp_norm_type': 'unpreconditioned',
                     'ksp_richardson_self_scale': False,
                     'ksp_type': 'richardson',
                     'pc_type': 'mg',
                     'pc_mg_type': 'full',
                     'mg_coarse_assembled_pc_type': 'lu',
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'mumps',
                     'mg_coarse_assembled_mat_mumps_icntl_14': 1000,
                     'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
                     'mg_coarse_pc_type': 'python',
                     'mg_levels': {'ksp_convergence_test': 'skip',
                                   'ksp_max_it': 5,
                                   'ksp_type': 'fgmres',
                                   'pc_python_type': 'firedrake.ASMStarPC',
                                   'pc_type': 'python'},
                    },
    'fieldsplit_1': {'ksp_type': 'richardson',
                     'ksp_max_it': 1,
                     'ksp_convergence_test': 'skip',
                     'ksp_richardson_scale': 1,
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.MassInvPC',
                     'Mp_pc_type': 'jacobi'},
}


# Save solutions
(u_, p_) = w.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
pvd = VTKFile("output/navierstokes.pvd")

# Solve problem

for Re_ in [1, 100, 500] + list(range(1000, 5100, 500)):
    Re.assign(Re_)
    print(BLUE % f"Solving for Re = {Re_}")
    sp['fieldsplit_1']['ksp_richardson_scale'] = -(2/float(Re) + float(gamma))
    solve(F == 0, w, bcs, solver_parameters=sp)

    # Monitor incompressibility
    print(f"||div u||: {norm(div(u), 'L2'):.2e}")

    pvd.write(u_, p_)
