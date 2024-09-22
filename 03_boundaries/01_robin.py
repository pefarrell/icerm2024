from firedrake import *

mesh = UnitSquareMesh(64, 64, quadrilateral=True)
(x, y) = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

u = Function(V, name="Solution")
v = TestFunction(V)

f = Constant(1)
g = Constant(1)
bc = DirichletBC(V, Constant(1), (2,3,4))

u.interpolate(Constant(5))
F = (
      inner(grad(u), grad(v))*dx
    + inner(u, v)*ds
    - inner(f, v)*dx
    - inner(g, v)*ds
    )

solve(F == 0, u, bc)

VTKFile("output/robin.pvd").write(u)
