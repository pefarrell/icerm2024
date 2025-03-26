from firedrake import *

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
u = Function(V, name="Solution")           # u_n
u_prev = Function(V, name="PrevSolution")  # u_{n-1}
v = TestFunction(V)

# Initial condition
(x, y, z) = SpatialCoordinate(mesh)
g = sin(z) * cos(x)

T  = 1              # final time
t  = 0              # current time we are solving for
dt = Constant(0.02) # timestep

u.interpolate(g)  # assign initial guess for solver
u_prev.assign(u)  # assign initial condition to u_0

F = (
      1/dt * inner(u - u_prev, v)*dx
           + inner(grad(u), grad(v))*dx
    )

output = VTKFile("output/heat.pvd")
output.write(u, time=t)

# Main timestepping loop
while True:
    # Update the time we're solving for
    t += float(dt)
    print(f"Solving for time: {t:.2f}")

    solve(F == 0, u)

    # Now cycle the variables
    u_prev.assign(u)

    output.write(u, time=t)
    if t >= T:
        break
