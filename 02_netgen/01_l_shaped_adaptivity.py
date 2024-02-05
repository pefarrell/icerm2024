from firedrake import *
from netgen.occ import *

def solve_poisson(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    uh = Function(V, name="Solution")
    v = TestFunction(V)
    bc = DirichletBC(V, 0, "on_boundary")
    f = Constant(1)
    F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
    solve(F == 0, uh, bc)
    return uh


def estimate_error(mesh, uh):
    W = FunctionSpace(mesh, "DG", 0)
    eta_sq = Function(W)
    w = TestFunction(W)
    f = Constant(1)
    h = CellDiameter(mesh)  # symbols for mesh quantities
    n = FacetNormal(mesh)
    v = CellVolume(mesh)

    # Compute error indicator cellwise
    G = (
          inner(eta_sq / v, w)*dx
        - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
        )

    # Each cell is an independent 1x1 solve, so Jacobi is exact
    sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G == 0, eta_sq, solver_parameters=sp)
    eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2

    with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
        error_est = sqrt(eta_.dot(eta_))
    return (eta, error_est)


def adapt(mesh, eta):
    W = FunctionSpace(mesh, "DG", 0)
    markers = Function(W)

    # We decide to refine an element if its error indicator
    # is within a fraction of the maximum cellwise error indicator

    # Access storage underlying our Function
    # (a PETSc Vec) to get maximum value of eta
    with eta.dat.vec_ro as eta_:
        eta_max = eta_.max()[1]

    theta = 0.5
    should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
    markers.interpolate(should_refine)

    refined_mesh = mesh.refine_marked_elements(markers)
    return refined_mesh


rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmsh)

max_iterations = 10
error_estimators = []
dofs = []
for i in range(max_iterations):
    print(f"Solving on level {i}")

    uh = solve_poisson(mesh)
    File(f"output/adaptive_loop_{i}.pvd").write(uh)

    (eta, error_est) = estimate_error(mesh, uh)
    print(f"  ||u - u_h|| <= C x {error_est}")
    error_estimators.append(error_est)
    dofs.append(uh.function_space().dim())

    mesh = adapt(mesh, eta)

import matplotlib.pyplot as plt
import numpy as np

plt.grid()
plt.loglog(dofs, error_estimators, '-ok', label="Measured convergence")
scaling = 1.5 * error_estimators[0]/dofs[0]**-(0.5)
plt.loglog(dofs, np.array(dofs)**(-0.5) * scaling, '--', label="Optimal convergence")
plt.xlabel("Number of degrees of freedom")
plt.ylabel("Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$")
plt.legend()
plt.savefig("adaptive_convergence.png")
#plt.show()
