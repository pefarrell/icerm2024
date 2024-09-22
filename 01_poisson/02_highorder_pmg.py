from firedrake import *

mesh = UnitSquareMesh(5, 5, quadrilateral=True)

(x, y) = SpatialCoordinate(mesh)
u_ex = sin(3*x) * exp(x + y)
f = -div(grad(u_ex))

p = 8
V = FunctionSpace(mesh, "Lagrange", p)
print(f"# of degrees of freedom: {V.dim()}")

u = Function(V, name="Solution")
v = TestFunction(V)

bc = DirichletBC(V, u_ex, "on_boundary")

F = inner(grad(u), grad(v))*dx - f*v*dx


sp = {
    "mat_type": "matfree",
    "ksp_type": "cg",
    "ksp_monitor": None,
    "ksp_rtol": 1.0e-14,
    "pc_type": "python",
    "pc_python_type": "firedrake.P1PC",
    "pmg_mg_coarse": {
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "cholesky",
    },
    "pmg_mg_levels": {
        "ksp_max_it": 1,
        "ksp_type": "chebyshev",
        "pc_type": "python",
        "pc_python_type": "firedrake.FDMPC",
        "fdm": {
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMExtrudedStarPC",
            "pc_star_mat_ordering_type": "metisnd",
            "pc_star_sub_sub_pc_type": "cholesky",
        }
    }
}

solve(F == 0, u, bc, solver_parameters=sp)
VTKFile("output/highorder.pvd").write(u)

print(f"||u - u_ex||_H1: {norm(u - u_ex, 'H1')}")
