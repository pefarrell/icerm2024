from firedrake import *
from collections.abc import Iterable
from numpy import linspace

_print = print
def print(x):
    if COMM_WORLD.rank == 0:
        _print(x, flush=True)

# Use a triangular mesh
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
base = UnitSquareMesh(16, 16, diagonal="crossed", distribution_parameters=distribution_parameters)
mh = MeshHierarchy(base, 1)
mesh = mh[-1]
n = FacetNormal(mesh)
(x, y) = SpatialCoordinate(mesh)

# Define function space
k = 4
V = FunctionSpace(mesh, "BDM", k)
Q = FunctionSpace(mesh, "DG", k-1, variant="integral")
W = MixedFunctionSpace([V, Q])

# Define Reynolds number and bcs
Re = Constant(1)
bcs = [DirichletBC(W.sub(0), Constant((0, 0)), (1, 2, 3)),
       DirichletBC(W.sub(0), as_vector([16 * x**2 * (1-x)**2, 0]), (4,))]

w = Function(W, name="Solution")
(u, p) = split(w)
z = TestFunction(W)
(v, q) = split(z)

# Augmented Lagrangian term
gamma = Constant(10000)

# DG penalty term
sigma = Constant(5 * (k+1)**2)
h = CellDiameter(mesh)
uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u

F = (
      2/Re                 * inner(sym(grad(u)), sym(grad(v)))*dx
    - 1/Re                 * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n)))*dS
    - 1/Re                 * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n)))*dS
    + 1/ Re * sigma/avg(h) * inner(2*avg(outer(u,n)),2*avg(outer(v,n)))*dS
    -                        inner(u ,div(outer(v,u)))*dx
    +                        inner(v('+')-v('-'), uflux_int('+')-uflux_int('-'))*dS
    -                        inner(p, div(v))*dx
    + gamma                * inner(div(u), div(v))*dx
    -                        inner(q, div(u))*dx
    )

def a_bc(u, v, bid, g):
    return (
           - 2/Re * inner(outer(v,n),sym(grad(u)))*ds(bid)
           - 2/Re             * inner(outer(u-g,n),sym(grad(v)))*ds(bid)
           + 1/Re * (sigma/h) * inner(v,u-g)*ds(bid)
           )

def c_bc(u, v, bid, g):
    if g is None:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
    else:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g
    return dot(v, uflux_ext)*ds(bid)

exterior_markers = set(mesh.exterior_facets.unique_markers)
for bc in bcs:
    if "DG" in str(bc._function_space):
        continue
    g = bc.function_arg
    bid = bc.sub_domain
    if isinstance(bid, Iterable):
        [exterior_markers.remove(_) for _ in bid]
    else:
        exterior_markers.remove(bid)
    F += a_bc(u, v, bid, g)
    F += c_bc(u, v, bid, g)
for bid in exterior_markers:
    F += c_bc(u, v, bid, None)

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
                     'mg_coarse_assembled_pc_factor_mat_solver_type': 'superlu_dist',
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
