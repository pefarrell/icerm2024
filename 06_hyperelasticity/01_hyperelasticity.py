from firedrake import *
from netgen.occ import *
from numpy import linspace

# Build 3x3 mesh of holes
rect = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,1).Face().bc("sides")
rect.edges.Min(Y).name = "bottom"
rect.edges.Max(Y).name = "top"

shape = rect
for i in range(1, 4):
    for j in range(1, 6):
        centre_x = 0.25*(j-1)
        centre_y = 0.25*(i+0)
        disk = WorkPlane(Axes((centre_x, centre_y, 0), n=Z, h=X)).Circle(0.1).Face()
        shape = shape - disk

geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=1)
base = Mesh(ngmesh)
mh = MeshHierarchy(base, 2, netgen_flags={})
mesh = mh[-1]
d = mesh.geometric_dimension()

bottom = [i + 1 for (i, name) in
         enumerate(ngmesh.GetRegionNames(codim=1)) if name == "bottom"]
top    = [i + 1 for (i, name) in
         enumerate(ngmesh.GetRegionNames(codim=1)) if name == "top"]

V = VectorFunctionSpace(mesh, "CG", 2)
u = Function(V, name="Displacement")

# Kinematics
I = Identity(d)  # Identity tensor
F = I + grad(u)  # Deformation gradient
C = F.T*F        # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
mu = Constant(400000)
lmbda = Constant(600000)
print(f"μ: {float(mu)}")
print(f"λ: {float(lmbda)}")

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - d) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
J = psi*dx

# Hyperelasticity equations. Quite hard to write down!
R = derivative(J, u)

# Boundary conditions
strain = Constant(0)
bcs = [DirichletBC(V, Constant((0, 0)), bottom),
       DirichletBC(V.sub(0), 0, top),
       DirichletBC(V.sub(1), strain, top)]

sp = {"snes_monitor": None, "snes_linesearch_type": "l2"}
#sp = {"snes_monitor": None}
pvd = File("output/hyperelasticity.pvd")
pvd.write(u, time=0)
for strain_ in linspace(0, -0.1, 41)[1:]:
    print(f"Solving for strain {strain_:.4f}")
    strain.assign(strain_)
    solve(R == 0, u, bcs, solver_parameters=sp)
    pvd.write(u, time=-strain_)
