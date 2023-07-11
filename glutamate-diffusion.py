from dolfin import (Expression, Constant, FunctionSpace, TrialFunction, TestFunction,
 Measure, inner, grad, interpolate, Function, assemble, solve)

from neuralthreesome.meshprocessing import hdf2fenics, geo2hdf
from neuralthreesome.timeseriesstorage import TimeSeriesStorage

def get_vesicle(ti, oi, parameters):
    m = 1.
    w = 0.5
    a = m / w
    sigma_t = parameters["sigma_t"]
    return Expression(
        "abs(x[1] - oi) <= half_w ? a * exp(-pow(t - ti, 2) / (2*sigma_t*sigma_t)) : 0.",
        ti=Constant(ti), oi=Constant(oi), a=Constant(a), sigma_t=Constant(sigma_t),
        t=Constant(0.), half_w=Constant(w/2), degree=0
    )
    
def vesicle_variational(vesicles, v, ds):
    terms = []
    for ves_i in vesicles:
        terms.append(single_vesicle_variational(ves_i, v, ds))
    return sum(terms)

def single_vesicle_variational(vesicle, v, ds):
    return vesicle * v * ds


geofile = "../resources/square.geo"
meshfile = "../resources/square.h5"

def glutamate_diffusion(dt, T, cell_size, vesicle_data, parameters, results_path, dirichlet_bcs=None):
    if dirichlet_bcs is None:
        bcs = []
    
    # Create and label meshes, TODO: Collect in object
    geo2hdf(geofile, meshfile, cell_size=cell_size, cleanup=False)
    mesh, _, boundaries = hdf2fenics(meshfile)
    boundary_labels = {
        "p": 1,
        "a": 2,
        "t": 3,
        "o": 4,
    }
    subdomain_labels = {
        "e": 1
    }
    
    # Define list of vesicles
    vesicles = [
        get_vesicle(ti, oi, parameters["vesicle"]) for ti, oi in vesicle_data
    ]
    
    
    # Define functionspaces, constants, etc. needed for var form 
    V = FunctionSpace(mesh, "CG", 1)
    g  = TrialFunction(V)
    v = TestFunction(V)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    D = parameters["D"]
    k_t = parameters["k_t"]
    k_a = parameters["k_a"]
    g0 = interpolate(Constant(0.0), V)  # Initial condition
    
    a = ((g * v + dt * inner(D * grad(g), grad(v))) * dx
         + dt * (k_t * g * v * ds(boundary_labels["t"]) + k_a * g * v * ds(boundary_labels["a"])))
    L = g0 * v * dx
    L += dt * vesicle_variational(vesicles, v, ds(boundary_labels["p"]))
    
    # Object responsible for writing results to HDF5-File
    storage = TimeSeriesStorage("w", results_path, mesh=mesh, V=V)
    
    g = Function(V)
    t = Constant(0.)
    A = assemble(a)  # Pre-assemble matrix.
    while float(t) < T:
        b = assemble(L)
        for bc in bcs:
            bc.apply(A, b)
        solve(A, g.vector(), b)
        storage.write(g, t)
        
        # Update time, prepare for next step
        g0.assign(g)
        for ves_i in vesicles:
            ves_i.t.assign(t + dt)
        t.assign(t + dt)
    
    storage.close()
    visual = TimeSeriesStorage("r", results_path)
    visual.to_xdmf(["glutamate"])
    visual.close()
    print("Done")