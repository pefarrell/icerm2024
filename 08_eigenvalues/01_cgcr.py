from firedrake import *
import numpy as np

errors_cg = []
errors_cr = []
target_eigenvalue = 9

# exact eigenvalues are n^2 + m^2, n, m \in \mathbb{N}
exact_eigenvalues = [2, 5, 5, 8, 10, 10, 13, 13, 17, 17, 18, 20, 20, 25, 25, 26, 26]


for N in [50, 100, 200]:
    mesh = RectangleMesh(N, N, pi, pi)

    for space in ["CG", "CR"]:
        V = FunctionSpace(mesh, space, 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx
        b = inner(u, v)*dx
        bc = DirichletBC(V, 0, "on_boundary")
        eigenproblem = LinearEigenproblem(a, b, bc)

        sp = {"eps_gen_hermitian": None,  # kind of problem
              "eps_smallest_real": None,  # which eigenvalues
              "eps_monitor": None,        # monitor
              "eps_type": "krylovschur"}  # algorithm

        # request ten eigenvalues
        eigensolver = LinearEigensolver(eigenproblem, 10, solver_parameters=sp)
        nconv = eigensolver.solve()  # number of converged eigenvalues

        # Take real part, since we know it is Hermitian
        eigenvalues = [eigensolver.eigenvalue(i).real for i in range(nconv)]
        print(f"{space}/{N}. Eigenvalues: ", eigenvalues)
        # Only take real part; .eigenfunction returns (real, complex)
        eigenfuncs  = [eigensolver.eigenfunction(i)[0] for i in range(nconv)]

        if space == "CR":
            errors_cr.append(eigenvalues[target_eigenvalue] - exact_eigenvalues[target_eigenvalue])
        elif space == "CG":
            errors_cg.append(eigenvalues[target_eigenvalue] - exact_eigenvalues[target_eigenvalue])



convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
print(f"Convergence orders (CG)", convergence_orders(errors_cg))
print(f"Convergence orders (CR)", convergence_orders(errors_cr))

