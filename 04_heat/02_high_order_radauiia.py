from firedrake import *
from irksome import *

# Parametric domain and coordinates
p_mesh = PeriodicRectangleMesh(64, 64, 4*pi, 2*pi, quadrilateral=True)
(u, v) = SpatialCoordinate(p_mesh)

# Parameters for Klein manifold
(a, n, m) = (2, 2, 1)

# Coordinate transformation
x = (a + cos(n*u/2.0) * sin(v) - sin(n*u/2.0) * sin(2*v)) * cos(m*u/2.0)
y = (a + cos(n*u/2.0) * sin(v) - sin(n*u/2.0) * sin(2*v)) * sin(m*u/2.0)
z = sin(n*u/2.0) * sin(v) + cos(n*u/2.0) * sin(2*v)

# Interpolate the coordinates into a vector field
V = VectorFunctionSpace(p_mesh, "CG", 3, dim=3)
coords = Function(V)
coords.interpolate(as_vector([x, y, z]))

# Make a mesh, using the topology of the base mesh,
# with coordinates from the supplied vector field
mesh = Mesh(coords)

# mesh = ...

V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="Solution")  # u_n
v = TestFunction(V)

# Initial condition
(x, y, z) = SpatialCoordinate(mesh)
g = sin(z) * cos(x)
u.interpolate(g)  # assign initial condition

F = (
      inner(Dt(u), v)*dx
    + inner(grad(u), grad(v))*dx
    )

# Irksome setup. RadauIIA is L-stable. RadauIIA(1) is backward Euler.
tableau = RadauIIA(1)
factory = MeshConstant(mesh)  # constants in space, not in time
dt = factory.Constant(0.02)   # timestep
t = factory.Constant(0)       # current time value
stepper = TimeStepper(F, tableau, t, dt, u)
T  = 1  # final time

output = File("output/irksome_heat.pvd")
output.write(u, time=float(t))

# Main timestepping loop
while True:
    # Solve for the next timestep
    print(f"Solving for time: {float(t + dt):.2f}")
    stepper.advance()  # derive and solve RK system
    t.assign(t + dt)

    output.write(u, time=float(t))
    if float(t) >= T:
        break
