from firedrake import *

mesh = UnitSquareMesh(10, 10, quadrilateral=True)

(x, y) = SpatialCoordinate(mesh)
u_ex = sin(3*x) * exp(x + y)
f = -div(grad(u_ex))

V = FunctionSpace(mesh, "Lagrange", 1)

u = Function(V, name="Solution")
v = TestFunction(V)

bc = DirichletBC(V, u_ex, "on_boundary")

F = inner(grad(u), grad(v))*dx - f*v*dx

solve(F == 0, u, bc)
VTKFile("output/poisson.pvd").write(u)

print(f"||u - u_ex||_H1: {norm(u - u_ex, 'H1')}")
