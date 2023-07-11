import sys
from time import sleep
from typing import Dict, List

import numpy as np
from dolfin import (
    Constant,
    DirichletBC,
    FiniteElement,
    Function,
    FunctionSpace,
    HDF5File,
    Measure,
    MixedElement,
    MixedFunctionSpace,
    MixedLinearVariationalProblem,
    PETScLUSolver,
    PETScMatrix,
    PETScNestMatrix,
    PETScVector,
    TestFunctions,
    TrialFunctions,
    Vector,
    XDMFFile,
    as_backend_type,
    assemble_mixed,
    assign,
    exp,
    extract_blocks,
    grad,
    inner,
    interpolate,
    ln,
    project,
)
from dolfin.function.argument import Argument
from petsc4py import PETSc
from scipy.sparse import bmat, csr_matrix
from tqdm import tqdm

from neuralthreesome.subdomains import SubMeshCollection


class SynapseSolver:
    def __init__(
        self,
        domain: SubMeshCollection,
        ion_list: List[Dict[str, float]],
        time_constant, # TODO: Figure out if needed
        lagrange_tag: str,
        **params
    ):
        self.ions = ion_list
        self.params = params
        self.domain = domain
        self.bcs = []
        self.dt = self.params['dt']
        self.t = time_constant


        self._create_functionspaces(ion_list, domain, lagrange_tag)

    def _create_functionspaces(
        self, ion_list, domain: SubMeshCollection, lagrange_tag: str = "exterior"
    ):
        K = len(ion_list)
        self.N_ions = K

        # define elements
        ufl_cell_subdomains = domain.subdomains[0].ufl_cell()
        ufl_cell_interfaces = domain.interfaces[0].ufl_cell()
        P1 = FiniteElement(
            "P", ufl_cell_subdomains, 1
        )  # ion concentrations and potentials
        Q1 = FiniteElement("P", ufl_cell_interfaces, 1)  # total membrane current I_M
        R0 = FiniteElement(
            "R", ufl_cell_subdomains, 0
        )  # Lagrange to enforce /int phi_i = 0

        subdomain_elements = [
            [P1] * (K + 1) + (lagrange_tag == subdomain.name()) * [R0]
            for subdomain in domain.subdomains
        ]
        self.Wr = {
            subdomain.name(): FunctionSpace(subdomain, MixedElement(el))
            for subdomain, el in zip(domain.subdomains, subdomain_elements)
        }
        self.Wg = {
            iface.name(): FunctionSpace(iface, Q1) for iface in domain.interfaces
        }
        self.W = MixedFunctionSpace(
            *[self.Wr[sub.name()] for sub in domain.subdomains],
            *[self.Wg[iface.name()] for iface in domain.interfaces]
        )
        self.iface_to_region_map = {}

    def create_variational_form(
        self, domain: SubMeshCollection, lagrange_tag: str = "exterior"
    ):
        # TODO: Extend to allow for dirichlet boundary conditions
        params = self.params
        dt = params["dt"]  # global time step (s)
        dt_inv = Constant(1.0 / dt)  # invert global time step (1/s)
        F = params["F"]  # Faraday's constant (C/mol)
        R = params["R"]  # Gas constant (J/(K*mol))
        temperature = params["temperature"]  # temperature (K)
        psi = R * temperature / F  # help variable psi
        C_M = params["C_M"]  # capacitance (F)
        phi0 = params["phi_M_init"]  # initial membrane potential (V)

        self.phi0 = {
            interface.name(): interpolate(phi0, self.Wg[interface.name()])
            for interface in domain.interfaces
        }

        dX = {sub.name(): sub.get_measure() for sub in domain.subdomains}
        dG = {iface.name(): iface.get_measure() for iface in domain.interfaces}


        # Define Test and TrialFunctions
        u, I = unpack_function_space(TrialFunctions(self.W), self)
        v, q = unpack_function_space(TestFunctions(self.W), self)

        # Define solution function
        self.u0 = Function(self.W)
        u0 = {
            sub.name(): self.u0.sub(idx) for idx, sub in enumerate(domain.subdomains)
        }

        # Define potential Trial and testfunctions.
        phi = {sub.name(): u[sub.name()][self.N_ions] for sub in domain.subdomains}
        w = {sub.name(): v[sub.name()][self.N_ions] for sub in domain.subdomains}

        # Define lagrange multiplier functions.
        _c = u[lagrange_tag][-1]
        _d = v[lagrange_tag][-1]

        # Initialize stuff
        I_ch = {interface.name(): 0. for interface in domain.interfaces}
        alpha_sum = {
            interface.name(): {
                interface.inside: 0.0,
                interface.outside: 0.0,
            }
            for interface in domain.interfaces
        }
        J = {sub.name(): 0.0 for sub in domain.subdomains}

        # Initialize parts of variational form.
        # TODO: Collect into smaller functions.
        for region in map(lambda x: x.name(), domain.subdomains):
            for idx, ion in enumerate(self.ions):
                z = ion.z
                D = ion.D
                # Set intial value of regional ion concentrations
                assign(
                    u0[region].sub(idx),
                    interpolate(
                        ion.k_init[region], self.Wr[region].sub(idx).collapse()
                    ),
                )

        for interface in domain.interfaces:
            for idx, ion in enumerate(self.ions):
                z = ion.z
                D = ion.D
                name = interface.name()
                u0g_i = interpolate(u0[interface.inside].sub(idx), self.Wg[name])
                u0g_e = interpolate(u0[interface.outside].sub(idx), self.Wg[name])
                alpha_sum[name][interface.inside] += D * z * z * u0g_i
                alpha_sum[name][interface.outside] += D * z * z * u0g_e
                # TODO: Identify what is meant by the two todo's below.
                # TODO: add subdomain information (not sure exactly what happens here)
                # TODO: Define a function to map Nernst Potential onto membrane
                ion.E[name] = project(psi / z * ln(u0g_e / u0g_i), self.Wg[name])
                ion.I_ch[name] = self.phi0[name]
                I_ch[name] += ion.I_ch[name]

        # TODO: Initialize already here that all blocks are involved.
        self.a = 0.0
        self.L = 0.0

        # Add lagrange multipliers
        self.a += (_c * w[lagrange_tag] + _d * phi[lagrange_tag]) * dX[lagrange_tag]

        # TODO: Collect below loops into, the following function.
        # J = add_concentration_weak_forms(self.a, self.L, domain, ions)
        for subdomain in domain.subdomains:
            region = subdomain.name()
            for idx, ion in enumerate(self.ions):
                z = ion.z
                D = ion.D
                I_ch_k = ion.I_ch
                k = u[region][idx]
                k0 = u0[region][idx]
                vk = v[region][idx]

                Jk_r = -Constant(D) * grad(k) - Constant(D * z / psi) * k0 * grad(phi[region])
                J[region] += F * z * Jk_r

                # Add interior terms of convection diffusion equation for kr
                self.a += (dt_inv * k * vk - inner(Jk_r, grad(vk))) * dX[region]

                # Loop over interfaces to add membrane currents
                for iface in subdomain.interfaces:
                    name = iface.name()
                    o = iface.orientation(region)

                    u0g = interpolate(u0[region].sub(idx), self.Wg[name])
                    alpha = D * z * z * u0g / alpha_sum[name][region]
                    self.a += o * (alpha * I[name] / (F * z)) * vk * dG[name]
                    self.L += (
                        (dt_inv * k0 * vk) * dX[region] 
                        - o * (I_ch_k[name] - alpha * I_ch[name]) / (F * z) * vk * dG[name]
                    )

        # Add weak form potentials.
        for subdomain in domain.subdomains:
            region = subdomain.name()
            self.a += inner(J[region], grad(w[region])) * dX[region]
            for iface in domain.interfaces:
                o = iface.orientation(region)
                self.a += o * inner(I[iface.name()], w[region]) * dG[iface.name()]

        # Add membrane current weak forms
        # TODO: Abstract into function
        # TODO: Investigate knpemi-solver line 248 on splitting schemes.
        for interface in domain.interfaces:
            name = interface.name()
            # o = interface.orientation(region)
            self.a += (
                    inner(phi[interface.inside] - phi[interface.outside] - (dt / C_M) * I[name], q[name]) * dG[name]
            )
            self.L += inner(self.phi0[name], q[name]) * dG[name]

        # TODO: Add hack for non-empty blocks
        fill_blocks_hack(self)

        self.wh = Function(self.W)
        return


    def solve_for_time_step(self):
        """ solve system for one global time step dt """
        # output to terminal
        dt = self.params['dt']                   # global time step (s)
        F = self.params['F']                     # Faraday's constant (C/mol)
        R = self.params['R']                     # Gas constant (J/(K mol))
        temperature = self.params['temperature'] # temperature (K)

        # reassemble the block that change in time
        self.matrix_blocks[2] = assemble_mixed(self.alist[2])
        self.matrix_blocks[6] = assemble_mixed(self.alist[6])
        self.matrix_blocks[3] = assemble_mixed(self.alist[3])
        self.matrix_blocks[7] = assemble_mixed(self.alist[7])
        # assemble right hand side
        Llist = extract_blocks(self.L)
        rhs_blocks = [assemble_mixed(L) for L in Llist]# if L is not None]

        AA = PETScNestMatrix(self.matrix_blocks)
        bb = PETScVector()
        AA.init_vectors(bb, rhs_blocks)
        # Convert VECNEST to standard vector for LU solver (MUMPS doesn't like VECNEST)
        bb = PETScVector(PETSc.Vec().createWithArray(bb.vec().getArray()))
        # Convert MATNEST to AIJ for LU solver
        AA.convert_to_aij()

        comm = self.domain.subdomains[0].mpi_comm()
        w = Vector(comm, sum([Wi.dim() for Wi in self.W.sub_spaces()]))

        solver = PETScLUSolver()        # create LU solver
        ksp = solver.ksp()              # get ksp  solver
        pc = ksp.getPC()                # get pc
        pc.setType("lu")                # set solver to LU
        pc.setFactorSolverType("mumps") # set LU solver to use mumps

        opts = PETSc.Options()          # get options
        opts["mat_mumps_icntl_4"] = 1   # set amount of info output
        opts["mat_mumps_icntl_14"] = 40 # set percentage of ???
        ksp.setFromOptions()            # update ksp with options set above

        solver.solve(AA, w, bb)             # solve system        for 


        # Assign the obtained solution to function wh defined on the FunctionSpaceProduct
        start = 0
        for i, Wi in enumerate(self.W.sub_spaces()):
            stop = start + Wi.dim()
            wi = Function(Wi).vector()
            wi.set_local(w.get_local()[start:stop])
            wi.apply('insert')
            self.wh.sub(0).assign(Function(Wi, wi))
            start = stop
            self.u0.sub(i).assign(self.wh.sub(i))

        for iface in self.domain.interfaces:
            idx_inside = [sub.name() for sub in self.domain.subdomains].index(iface.inside)
            idx_outside = [sub.name() for sub in self.domain.subdomains].index(iface.outside)
            self.phi0[iface.name()].assign(
                (interpolate(self.u0.sub(idx_inside).sub(self.N_ions), self.Wg[iface.name()])
                -interpolate(self.u0.sub(idx_outside).sub(self.N_ions), self.Wg[iface.name()]))
            )
            # update Nernst potential for all ions
            for idx, ion in enumerate(self.ions):
                z = ion.z
                ki_prev_g = interpolate(self.u0.sub(idx_inside).sub(idx), self.Wg[iface.name()])
                ke_prev_g = interpolate(self.u0.sub(idx_outside).sub(idx), self.Wg[iface.name()])
                ion.E[iface.name()].assign(project(R*temperature/(F*z)*ln(ke_prev_g/ki_prev_g), self.Wg[iface.name()]))
                # TODO: Insert psi in above expression

        # # update previous membrane potential
        # self.phi_M_prev.assign(interpolate(self.u_p.sub(0).sub(self.N_ions), self.Wg) \
        #                      - interpolate(self.u_p.sub(1).sub(self.N_ions), self.Wg))

        # updates problems time t
        self.t.assign(float(self.t + dt))

        # # update Nernst potential for all ions
        # for idx, ion in enumerate(self.ion_list):
        #     z = ion['z']
        #     ki_prev_g = interpolate(self.u_p.sub(0).sub(idx), self.Wg)
        #     ke_prev_g = interpolate(self.u_p.sub(1).sub(idx), self.Wg)
        #     ion['E'].assign(project(R*temperature/(F*z)*ln(ke_prev_g/ki_prev_g), self.Wg))
        return

    def solve_system_passive(self, filename, dirichlet_bcs=False):
        """ solve KNP-EMI with passive dynamics on membrane """

        # create variational formulation
        # self.create_variational_form(dirichlet_bcs=dirichlet_bcs)
        self.create_variational_form(self.domain)

        # extract the subforms corresponding to each blocks of the formulation
        self.alist = extract_blocks(self.a)
        self.Llist = extract_blocks(self.L)

        # build the variational problem : build needed mappings and sort the BCs
        MixedLinearVariationalProblem(self.alist, self.Llist, self.wh.split(), self.bcs)

        # assemble all the blocks
        self.matrix_blocks = [assemble_mixed(a) for a in self.alist]
        self.rhs_blocks = [assemble_mixed(L) for L in self.Llist]

        # initialize save results
        self.initialize_h5_savefile(filename + '/results.h5')
        self.initialize_xdmf_savefile(filename + '/') # TODO: Create this.

        # global end time
        Tstop = self.params['Tstop']

        # solve
        sys.stdout.write('\n')
        for k in range(int(round(Tstop/float(self.dt)))):
            # solve for one time step
            self.solve_for_time_step()
            # save results
            self.save_h5()
            self.save_xdmf()
            # output to terminal
            mult = 100/int(round(Tstop/float(self.dt)))
            sys.stdout.write('\r')
            sys.stdout.write('progress: %d%%' % (mult*k))
            sys.stdout.flush()
            sleep(0.25)

        sys.stdout.write('\n')

        # close files
        self.close_h5()
        self.close_xdmf()
        return

    def solve_system_HH(self, n_steps_ode, filename, dirichlet_bcs=False):
        """ solve KNP-EMI with Hodgkin Huxley (HH) dynamics on membrane using a
            two-step splitting scheme """
        # physical parameters
        C_M = self.params['C_M']           # capacitance (F/m)
        g_Na_bar = self.params['g_Na_bar'] # Na conductivity HH (S/m^2)
        g_K_bar = self.params['g_K_bar']   # K conductivity HH (S/m^2)
        
        # initial values
        n_init = self.params['n_init']     # gating variable n
        m_init = self.params['m_init']     # gating variable m
        h_init = self.params['h_init']     # gating variable h
        V_rest = self.params['V_rest']     # resting potential (V)

        # Hodgkin Huxley parameters
        n = interpolate(Constant(n_init), self.Wg)
        m = interpolate(Constant(m_init), self.Wg)
        h = interpolate(Constant(h_init), self.Wg)

        # get Na of K from ion list
        Na = self.ion_list[0]
        K = self.ion_list[1]

        # add membrane conductivity of Hodgkin Huxley channels
        Na['g_k'] += g_Na_bar*m**3*h
        K['g_k'] += g_K_bar*n**4

        # create variational formulation
        self.create_variational_form(splitting_scheme=True, dirichlet_bcs=dirichlet_bcs)

        # extract the subforms corresponding to each blocks of the formulation
        self.alist = extract_blocks(self.a)
        self.Llist = extract_blocks(self.L)
        # build the variational problem : build needed mappings and sort the BCs
        MixedLinearVariationalProblem(self.alist, self.Llist, self.wh.split(), self.bcs)
        # assemble all the blocks
        self.matrix_blocks = [assemble_mixed(a) for a in self.alist]
        self.rhs_blocks = [assemble_mixed(L) for L in self.Llist]

        # convert phi_M from V to mV
        V_M = 1000*(self.phi_M_prev - V_rest)
        # rate coefficients
        alpha_n = 0.01e3*(10.-V_M)/(exp((10.-V_M)/10.) - 1.)
        beta_n = 0.125e3*exp(-V_M/80.)
        alpha_m = 0.1e3*(25. - V_M)/(exp((25. - V_M)/10.) - 1)
        beta_m = 4.e3*exp(-V_M/18.)
        alpha_h = 0.07e3*exp(-V_M/20.)
        beta_h = 1.e3/(exp((30.-V_M)/10.) + 1)

        # shorthand
        phi_M = self.phi_M_prev
        # derivatives for Hodgkin Huxley ODEs
        dphidt = - (1/C_M)*(Na['g_k']*(phi_M - Na['E']) + K['g_k']*(phi_M - K['E']))
        dndt = alpha_n*(1 - n) - beta_n*n
        dmdt = alpha_m*(1 - m) - beta_m*m
        dhdt = alpha_h*(1 - h) - beta_h*h

        # initialize saving of results
        save_count = 0
        self.initialize_h5_savefile(filename + 'results.h5')
        self.initialize_xdmf_savefile(filename)

        dt_ode = self.dt/n_steps_ode # ODE time step (s)
        Tstop = self.params['Tstop'] # global end time (s)
        # solve
        for k in range(int(round(Tstop/float(self.dt)))):
            # Step I: Solve Hodgkin Hodgkin ODEs using backward Euler
            for i in range(n_steps_ode):
                phi_M_new = project(phi_M + dt_ode*dphidt, self.Wg)
                n_new = project(n + dt_ode*dndt, self.Wg)
                m_new = project(m + dt_ode*dmdt, self.Wg)
                h_new = project(h + dt_ode*dhdt, self.Wg)
                assign(phi_M, phi_M_new)
                assign(n, n_new)
                assign(m, m_new)
                assign(h, h_new)

            # Step II: Solve PDEs with phi_M_prev from ODE step
            self.solve_for_time_step()

            # save results
            save_count += 1
            if save_count == 1:
                self.save_h5()
                self.save_xdmf()
                save_count = 0

            # output to terminal
            mult = 100/int(round(Tstop/float(self.dt)))
            sys.stdout.write('\r')
            sys.stdout.write('progress: %d%%' % (mult*k))
            sys.stdout.flush()
            sleep(0.25)

        # close results files
        self.close_h5()
        self.close_xdmf()
        return

    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx = 0
        self.h5_file = HDF5File(self.domain.subdomains[0].mpi_comm(), filename, 'w')
        self.h5_file.write(self.domain.mesh, '/mesh')
        self.h5_file.write(self.domain.cellfunction, '/subdomains')
        self.h5_file.write(self.domain.facetfunction, '/surfaces')
        for iface in self.domain.interfaces:
            self.h5_file.write(iface, "/interface_{}".format(iface.name()))
        
        self.save_h5()
        return

    def save_h5(self):
        """ save results to h5 file """
        for idx, sub in enumerate(self.domain.subdomains):
            self.h5_file.write(self.u0.sub(idx), '/{}_solution'.format(sub.name()),  self.h5_idx)
            
        for name in self.phi0:
            self.h5_file.write(self.phi0[name], "/{}_potential".format(name), self.h5_idx)
        
        self.h5_idx += 1
        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()
        return

    

    def initialize_xdmf_savefile(self, file_prefix):
        """ initialize xdmf file """
        self.subdomain_xdmf_files = {}
        ion_list_hack = [{'name': ion.name} for ion in self.ions] + [{'name': 'phi'}]
        for sub_idx, sub in enumerate(self.domain.subdomains):
            self.subdomain_xdmf_files[sub.name()] = []
            for idx, ion in enumerate(ion_list_hack):
                filename_xdmf = file_prefix + sub.name() + '_' +  ion['name'] + '.xdmf'
                xdmf_file = XDMFFile(sub.mpi_comm(), filename_xdmf)
                xdmf_file.parameters['rewrite_function_mesh'] = False
                xdmf_file.parameters['flush_output'] = True
                # TODO: Verify correctness of below; might need "True" as argument in .split()
                xdmf_file.write(self.u0.sub(sub_idx).split()[idx], self.t.values()[0])
                self.subdomain_xdmf_files[sub.name()].append(xdmf_file)
        
        self.interface_xdmf_files = []
        for iface in self.domain.interfaces:
            filename_xdmf = file_prefix + iface.name() + '_potential' + '.xdmf'
            iface_xdmf_file = XDMFFile(iface.mpi_comm(), filename_xdmf)
            iface_xdmf_file.parameters['rewrite_function_mesh'] = False
            iface_xdmf_file.parameters['flush_output'] = True
            iface_xdmf_file.write(self.phi0[iface.name()], self.t.values()[0])
            self.interface_xdmf_files.append(xdmf_file)

        return

    def save_xdmf(self):
        """ save results to xdmf files """
        for idx, sub in enumerate(self.domain.subdomains):

            for i in range(len(self.subdomain_xdmf_files[sub.name()])):
                self.subdomain_xdmf_files[sub.name()][i].write(self.u0.sub(idx).split()[i], self.t.values()[0])

        for idx, iface in enumerate(self.domain.interfaces):
            self.interface_xdmf_files[idx].write(self.phi0[iface.name()], self.t.values()[0])
        return

    def close_xdmf(self):
        """ close xdmf files """
        for filelist in self.subdomain_xdmf_files.values():
            for file in filelist:
                file.close()

        for file in self.interface_xdmf_files:
            file.close()
        return

    def collapse_to_PETCS(self, AA):
        """ collapse matrix to PETCS matrix """
        as_csr_matrix = lambda mat: csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
        mat_values = [as_csr_matrix(as_backend_type(A).mat())
                      if not isinstance(A, int) else None
                      for A in AA.blocks.flatten()]
        mat_values = np.array(mat_values).reshape(AA.blocks.shape)
        mat_values = bmat(mat_values).tocsr()

        AA = PETSc.Mat().createAIJ(size=mat_values.shape,
                                   csr=(mat_values.indptr, mat_values.indices, mat_values.data))
        AA = PETScMatrix(AA)
        return AA



def unpack_function_space(U: List[Argument], solver: SynapseSolver):
    u = {region: U[idx] for idx, region in enumerate(solver.Wr)}
    I = {interface: U[idx + len(solver.Wr)] for idx, interface in enumerate(solver.Wg)}
    return u, I


def fill_blocks_hack(solver):
    # Define Test and TrialFunctions
    u, I = unpack_function_space(TrialFunctions(solver.W), solver)
    v, q = unpack_function_space(TestFunctions(solver.W), solver)
    iface = solver.domain.interfaces[0]
    dG = Measure('dx', domain=iface)
    solver.a += Constant(0.) * u[iface.inside][0] * v[iface.outside][0] * dG
    solver.a += Constant(0.) * u[iface.outside][0] * v[iface.inside][0] * dG

    solver.a += Constant(0.) * I["post-terminal"] * q["post-membrane"] * dG
    solver.a += Constant(0.) * I["post-membrane"] * q["post-terminal"] * dG


