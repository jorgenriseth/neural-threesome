import os
import sys
from pathlib import Path
from typing import List

import h5py  # For some reason, needs to be imported first
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from dolfin.function.argument import Argument

from neuralthreesome.ion import Ion
from neuralthreesome.meshprocessing import geo2hdf, hdf2fenics
from neuralthreesome.parameters import default_parameters
from neuralthreesome.solver_knpemi import Solver as SolverKNPEMI
from neuralthreesome.solver_synapse import SynapseSolver, unpack_function_space
from neuralthreesome.subdomains import FacetDomainInfo, SubDomainInfo, SubMeshCollection


# Path to any results.
resultspath = Path('results/jorgen/synapse/')
resultspath.mkdir(parents=True, exist_ok=True)

# Path to gmsh-file, and where to output the resulting meshfile
geofile = "resources/synapse-simple.geo"
meshfile = "mesh/synapse-simple.h5"


# Create mesh from the gmsh-file
geo2hdf(geofile, meshfile, cell_size=0.1, cleanup=False)

# Load the mesh and the MeshFunctions tagging the different subdomains.
mesh, subdomain_tags, boundary_tags = hdf2fenics(meshfile)


# Information of subdomains.
subdomain_info = [
    SubDomainInfo(
        name="exterior",
        value=1,
        interfaces=["post-membrane", "post-terminal"],
        ext_bdrys=['empty', 'pre', 'astrocytes']
    ),
    SubDomainInfo(
        name="interior",
        value=2,
        interfaces=["post-membrane", "post-terminal"],
        ext_bdrys=["post-boundary"]
    )
]

# Add information about boundaries and interfaces.
boundary_info = [
    FacetDomainInfo("pre", 1),
    FacetDomainInfo("astrocytes", 2),
    FacetDomainInfo("post-membrane", 3, bordering_regions=("interior", "exterior")),
    FacetDomainInfo("post-terminal", 4, bordering_regions=("interior", "exterior")),
    FacetDomainInfo("post-boundary", 5),
    FacetDomainInfo("empty", 6)
]

# Create a collection of submeshes for the subdomains and boundaries.
domain = SubMeshCollection(subdomain_tags, boundary_tags, subdomain_info, boundary_info)

# How to access mesh of a specific subdomain
plot(domain.subdomains["exterior"])
plt.show()


from neuralthreesome.parameters import default_parameters

param = default_parameters()
Na = Ion(
    name="Na",
    valence=1.0,
    diffusion_coefficient=param["D_Na"],
    initial_concentrations={
        "interior": param["Na_i_init"],
        "exterior": param["Na_e_init"]
        
    }
)

K = Ion(
    name="K",
    valence=1.0,
    diffusion_coefficient=param["D_K"],
    initial_concentrations={
        "interior": param["K_i_init"],
        "exterior": param["K_e_init"]
        
    }
)

Cl = Ion(
    name="Cl",
    valence=-1.,
    diffusion_coefficient=param["D_Cl"],
    initial_concentrations={
        "interior": param["Cl_i_init"],
        "exterior": param["Cl_e_init"]
        
    }
)

ion_list = [Na, K, Cl]


# time variables (seconds)
dt = 1.0e-5                      # global time step (s)
Tstop = 1.0e-2                   # global end time (s)

# physical parameters
C_M = 0.02                       # capacitance (F)
temperature = 300                # temperature (K)
F = 96485                        # Faraday's constant (C/mol)
R = 8.314                        # Gas constant (J/(K*mol))
g_Na_leak = Constant(30*0.2)     # Na leak membrane conductivity (S/(m^2))
g_K_leak = Constant(30*0.8)      # K leak membrane conductivity (S/(m^2))
g_Cl_leak = Constant(0.0)        # Cl leak membrane conductivity (S/(m^2))
g_syn_bar = 1.25e3               # Na synaptic membrane conductivity (S/(m^2))
D_Na = Constant(1.33e-9)         # Na diffusion coefficient (m^2/s)
D_K = Constant(1.96e-9)          # K diffusion coefficient (m^2/s)
D_Cl = Constant(2.03e-9)         # Cl diffusion coefficient (m^2/s)

# EMI specific parameters
sigma_i = 2.01202                # intracellular conductivity
sigma_e = 1.31365                # extracellular conductivity
E_Na = 54.8e-3                   # reversal potential Na (V)
E_K = -88.98e-3                  # reversal potential K (V)
g_Na_leak_emi = Constant(30*0.2) # Na leak membrane conductivity (S/(m^2))
g_K_leak_emi = Constant(30*0.8)  # K leak membrane conductivity (S/(m^2))

# initial conditions
phi_M_init = Constant(-60e-3)    # membrane potential (V)
Na_i_init = Constant(12)         # intracellular Na concentration (mol/m^3)
Na_e_init = Constant(100)        # extracellular Na concentration (mol/m^3)
K_i_init = Constant(125)         # intracellular K concentration (mol/m^3)
K_e_init = Constant(4)           # extracellular K concentration (mol/m^3)
Cl_i_init = Constant(137)        # intracellular Cl concentration (mol/m^3)
Cl_e_init = Constant(104)        # extracellular Cl concentration (mol/m^3)

# set parameters
params = {'dt':dt, 'Tstop':Tstop,
          'temperature':temperature, 'R':R, 'F':F, 'C_M':C_M,
          'phi_M_init':phi_M_init,
          'sigma_i':sigma_i, 'sigma_e':sigma_e,
          'g_K_leak':g_K_leak_emi,
          'g_Na_leak':g_Na_leak_emi,
          'E_Na':E_Na, 'E_K':E_K}


solver = SynapseSolver(domain, ion_list, lagrange_tag="exterior", **params)
solver.create_variational_form(domain, "exterior")


alist = extract_blocks(solver.a)
Llist = extract_blocks(solver.L)
MixedLinearVariationalProblem(alist, Llist, solver.wh.split(), [])