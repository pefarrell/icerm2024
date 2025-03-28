from firedrake import *

base = UnitSquareMesh(10, 10, diagonal="crossed")
mh = MeshHierarchy(base, 3)
mesh = mh[-1]

(x, y) = SpatialCoordinate(mesh)
u_ex = sin(3*x) * exp(x + y)
f = -div(grad(u_ex))

p = 3
V = FunctionSpace(mesh, "Lagrange", p)
print(f"# of degrees of freedom: {V.dim()}")

u = Function(V, name="Solution")
v = TestFunction(V)

bc = DirichletBC(V, u_ex, "on_boundary")

F = inner(grad(u), grad(v))*dx - f*v*dx


sp = {
    "mat_type": "aij",
    "ksp_type": "cg",
    "ksp_view": None,
    "ksp_monitor": None,
    "ksp_rtol": 1.0e-14,
    "pc_type": "mg",
    "mg_coarse_pc_type": "cholesky",
    "mg_levels": {
        "ksp_max_it": 1,
        "ksp_type": "chebyshev",
        "pc_type": "jacobi",
    }
}

solve(F == 0, u, bc, solver_parameters=sp)
VTKFile("output/highorder.pvd").write(u)

print(f"||u - u_ex||_H1: {norm(u - u_ex, 'H1')}")
