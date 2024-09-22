from firedrake import *
from netgen.occ import *

# Mesh: a hole removed from a rectangle
disk = WorkPlane(Axes((0,0,0), n=Z, h=X)).Circle(1).Face()
rect = WorkPlane(Axes((-3,-3,0), n=Z, h=X)).Rectangle(13, 6).Face()
domain = rect - disk

# Label boundaries
domain.edges.name = "wall"  # all default to wall
domain.edges.Min(X).name = "inlet"
domain.edges.Max(X).name = "outlet"
geo = OCCGeometry(domain, dim=2)

ngmesh = geo.GenerateMesh(maxh=1)
base = Mesh(ngmesh)
mh = MeshHierarchy(base, 2, netgen_flags={"degree": 2})
mesh = mh[-1]

walls = [i + 1 for (i, name) in
         enumerate(ngmesh.GetRegionNames(codim=1)) if name == "wall"]

n = FacetNormal(mesh)
(x, y) = SpatialCoordinate(mesh)

# Define Taylor--Hood function space W
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, Q])

# Define Function and TestFunction(s)
w = Function(W, name="Solution")
(u, p) = split(w)
(v, q) = split(TestFunction(W))

# Define viscosity and bcs
nu = Constant(0.2)
p0 = 10/13 - x/13  # 1 at left, 0 at right
bcs = DirichletBC(W.sub(0), Constant((0, 0)), walls)

# Define variational form
F = (
      inner(2*nu*sym(grad(u)), sym(grad(v)))*dx
    - div(u)*q*dx
    - div(v)*p*dx
    + p0*dot(v,n)*ds
    )

# Solve problem
solve(F == 0, w, bcs)

# Save solutions
(u_, p_) = w.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")
VTKFile("output/stokes.pvd").write(u_, p_)
