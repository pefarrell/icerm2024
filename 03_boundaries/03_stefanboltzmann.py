from firedrake import *

mesh = UnitSquareMesh(64, 64, quadrilateral=True)
(x, y) = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

u = Function(V, name="Solution")
v = TestFunction(V)

f = Constant(1000)*x*(1-x)*y*(1-y)
c = Constant(0.5)

u.interpolate(Constant(5))
F = (
      inner(grad(u), grad(v))*dx
    - inner(f, v)*dx
    - inner(c**4 - u**4, v)*ds
    )

sp = {"snes_monitor": None,
      "snes_linesearch_type": "l2",
      "snes_linesearch_monitor": None}
solve(F == 0, u, solver_parameters=sp)

File("output/stefanboltzmann.pvd").write(u)
