from firedrake import *
from firedrake.petsc import PETSc
from math import pi

#Grabbing PETSc options
opts = PETSc.Options()
omega = opts.getReal("omega", 4.0)
k = opts.getReal("k", 10.0)
N = opts.getInt("N", 32)


mesh = UnitSquareMesh(N,N)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)


P = Function(V)
P = assemble(interpolate(sin(pi*omega*x), V))
P.rename("Exact")

F = Function(V)
F = assemble(interpolate(((pi*omega)**2)*sin(pi*omega*x)-(k**2)*sin(pi*omega*x), V))

#Solving the Helmholtz equation
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx - (omega**2)*inner(u,v)*dx
L = inner(F,v)*dx

p = Function(V)

bc = DirichletBC(V, P, "on_boundary")
solve(a == L, p, bcs=bc)
p.rename("Helmholtz")

F = Function(V)
F = assemble(interpolate(((pi*omega)**2)*sin(pi*omega*x), V))
#Solving the Poisson equation
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = inner(F,v)*dx

u = Function(V)

bc = DirichletBC(V, P, "on_boundary")
solve(a == L, u, bcs=bc)
u.rename("Poisson")

File("output/helmholtz.pvd").write(p, u, P)