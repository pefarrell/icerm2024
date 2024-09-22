from firedrake import *
from netgen.occ import *
from ngsPETSc import NetgenHierarchy

# Netgen uses string labels
cube = Box(Pnt(0,0,0), Pnt(1,1,1)).bc("cube")
sphere = Sphere(Pnt(1, 1, 1), 0.5).bc("sphere")

# We need to Glue them together to keep
# distinct surface labels
geo = OCCGeometry(Glue([cube, sphere]))
ngmesh = geo.GenerateMesh(maxh=0.5)

# Firedrake uses integer labels, so we need to convert.
cube_labels = [i + 1 for (i, name) in
            enumerate(ngmesh.GetRegionNames(codim=1)) if name == "cube"]
sphere_labels = [i + 1 for (i, name) in
             enumerate(ngmesh.GetRegionNames(codim=1)) if name == "sphere"]

# Make a high-order mesh of the union of cube and sphere
mh = NetgenHierarchy(ngmesh, 0, order=3)
mesh = mh[-1]

V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="Solution")
v = TestFunction(V)

# OK, this data is not H^{1/2}, but
# we use it for illustration anyway
bcs = [DirichletBC(V, +1, cube_labels),
       DirichletBC(V, -1, sphere_labels)]

F = inner(grad(u), grad(v))*dx
solve(F == 0, u, bcs)
VTKFile("output/bclabels.pvd").write(u)
