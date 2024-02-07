from firedrake import *

mesh = UnitSquareMesh(64, 64, quadrilateral=True)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V, name="Solution")
J = 0.5 * inner(grad(u), grad(u))*dx
F = derivative(J, u)
bcs = DirichletBC(V, 0, "on_boundary")

# Define obstacle
(x, y) = SpatialCoordinate(mesh)
r = 0.25  # radius
psi = 4*(r**2 - (x - 0.5)**2 - (y - 0.5)**2)
obstacle = Function(V, name="Obstacle")
obstacle.interpolate(conditional(lt(psi, 0), 0, psi))

# We have to use a slightly lower-level interface:
# under the hood, solve(F == 0) makes these objects
sp = {"snes_type": "vinewtonrsls",
      "snes_monitor": None}
problem = NonlinearVariationalProblem(F, u, bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

# Pass bounds in call to solver.solve: (lower bound, upper bound)
# Unfortunately we need to pass an upper bound, also.
upper = Function(V).interpolate(Constant(1e10))
solver.solve(bounds=(obstacle, upper))

File("output/obstacle.pvd").write(u, obstacle)
