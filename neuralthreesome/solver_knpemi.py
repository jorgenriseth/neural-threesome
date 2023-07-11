#!/usr/bin/python3

"""
Solves the KNP-EMI problem with either a passive membrane model or the Hodgkin
Huxley model.
"""

from dolfin import *
import numpy as np

from petsc4py import PETSc
from scipy.sparse import bmat, csr_matrix

from time import sleep
import sys

class Solver:
    def __init__(self, ion_list, t, MMS=None, **params):
        """ initialize solver """
        self.ion_list = ion_list    # set ion list
        self.params = params        # set parameters
        self.N_ions = len(ion_list) # set number of ions in ion list
        self.dt = params['dt']      # set global time step (s)
        self.MMS = MMS              # set MMS terms
        self.t = t                  # time constant (s)
        return

    def setup_domain(self, mesh, subdomains, surfaces):
        """ setup mortar meshes (ECS, ICS, gamma) and element spaces """
        # get and set mesh
        self.mesh = mesh
        # get and set subdomains
        self.subdomains = subdomains
        # get and set surfaces
        self.surfaces = surfaces

        # create meshes
        self.interior_mesh = MeshView.create(subdomains, 1) # interior mesh
        self.exterior_mesh = MeshView.create(subdomains, 0) # exterior mesh
        self.gamma_mesh = MeshView.create(surfaces, 1)      # interface mesh

        # mark exterior boundary of exterior mesh
        bndry_mesh = MeshView.create(surfaces, 2)    # create mesh for exterior boundary
        bndry_mesh.build_mapping(self.exterior_mesh) # create mapping between facets
        # access mapping
        facet_mapping = bndry_mesh.topology().mapping() [self.exterior_mesh.id()].cell_map()
        # create exterior boundary mesh function
        self.exterior_boundary = MeshFunction('size_t', self.exterior_mesh,
                                 self.exterior_mesh.topology().dim()-1, 0)
        # mark exterior facets with 2
        for i in range(len(facet_mapping)):
            self.exterior_boundary[facet_mapping[i]] = 2

        # define elements
        P1 = FiniteElement('P', self.interior_mesh.ufl_cell(), 1) # ion concentrations and potentials
        R0 = FiniteElement('R', self.interior_mesh.ufl_cell(), 0) # Lagrange to enforce /int phi_i = 0
        Q1 = FiniteElement('P', self.gamma_mesh.ufl_cell(), 1)    # total membrane current I_M

        # Intracellular (ICS) ion concentrations for each ion + ICS potential + Lagrange multiplier
        interior_element_list = [P1]*(self.N_ions + 1) + [R0]
        # Extracellular (ECS) ion concentrations for each ion + ECS potential
        exterior_element_list = [P1]*(self.N_ions + 1)

        # create function spaces
        self.Wi = FunctionSpace(self.interior_mesh, MixedElement(interior_element_list))
        self.We = FunctionSpace(self.exterior_mesh, MixedElement(exterior_element_list))
        self.Wg = FunctionSpace(self.gamma_mesh, Q1)
        self.W = MixedFunctionSpace(self.Wi, self.We, self.Wg)
        return

    def create_variational_form(self, splitting_scheme=False, dirichlet_bcs=False):
        """ create a mortar variation formulation for the KNP-EMI equations """
        params = self.params
        # get parameters
        dt = params['dt']                   # global time step (s)
        dt_inv = Constant(1./dt)            # invert global time step (1/s)
        F = params['F']                     # Faraday's constant (C/mol)
        R = params['R']                     # Gas constant (J/(K*mol))
        temperature = params['temperature'] # temperature (K)
        psi = R*temperature/F               # help variable psi
        C_M = params['C_M']                 # capacitance (F)
        phi_M_init = params['phi_M_init']   # initial membrane potential (V)

        # set initial membrane potential
        self.phi_M_prev = interpolate(phi_M_init, self.Wg)

        # define measures
        dxe = Measure('dx', domain=self.exterior_mesh)  # on exterior mesh
        dxi = Measure('dx', domain=self.interior_mesh)  # on interior mesh
        dxGamma = Measure('dx', domain=self.gamma_mesh) # on interface

        # measure on exterior boundary
        dsOuter = Measure('ds', domain=self.exterior_mesh,
                                subdomain_data=self.exterior_boundary,
                                subdomain_id=2)

        # define outward normal on exterior boundary (partial Omega)
        self.n_outer = FacetNormal(self.exterior_mesh)

        # MMS redefine measure on gamma (each side of interface uniquely marked)
        if self.MMS is not None:
            # # NB: Only holds for Omega_i = [0.25, 0.75] x [0.25, 0.75]
            gamma_subdomains = ('near(x[0], 0.25)', 'near(x[0], 0.75)',\
                                'near(x[1], 0.25)', 'near(x[1], 0.75)')
            # Mark interface
            gamma_boundary = MeshFunction('size_t', self.gamma_mesh, self.gamma_mesh.topology().dim(), 0)
            [subd.mark(gamma_boundary, i) for i, subd in enumerate(map(CompiledSubDomain, gamma_subdomains), 1)]  
            # redefine measure on gamma
            dxGamma = Measure('dx', domain=self.gamma_mesh, subdomain_data=gamma_boundary)

        # create functions
        (ui, ue, p_IM) = TrialFunctions(self.W)  # trial functions
        (vi, ve, q_IM) = TestFunctions(self.W)   # test functions
        self.u_p = Function(self.W)             # previous solutions
        ui_p = self.u_p.sub(0)
        ue_p = self.u_p.sub(1)

        # split unknowns
        ui = split(ui)
        ue = split(ue)
        # split test functions
        vi = split(vi)
        ve = split(ve)
        # split previous solution
        ui_prev = split(ui_p)
        ue_prev = split(ue_p)

        # intracellular potential
        phi_i = ui[self.N_ions]  # unknown
        vphi_i = vi[self.N_ions] # test function

        # Lagrange multiplier (int phi_i = 0)
        _c = ui[self.N_ions+1]   # unknown
        _d = vi[self.N_ions+1]   # test function

        # extracellular potential
        phi_e = ue[self.N_ions]  # unknown
        vphi_e = ve[self.N_ions] # test function

        # initialize
        alpha_i_sum = 0 # sum of fractions intracellular
        alpha_e_sum = 0 # sum of fractions extracellular
        I_ch = 0        # total channel current
        J_phi_i = 0     # total intracellular flux
        J_phi_e = 0     # total extracellular flux
        self.bcs = []   # Dirichlet boundary conditions

        # Initialize parts of variational formulation
        for idx, ion in enumerate(self.ion_list):
            # get ion attributes
            z = ion['z']; Di = ion['Di']; De = ion['De'];

            # set initial value of intra and extracellular ion concentrations
            assign(ui_p.sub(idx), interpolate(ion['ki_init'], self.Wi.sub(idx).collapse()))
            assign(ue_p.sub(idx), interpolate(ion['ke_init'], self.We.sub(idx).collapse()))

            # add ion specific contribution to fraction alpha
            ui_prev_g = interpolate(ui_p.sub(idx), self.Wg)
            ue_prev_g = interpolate(ue_p.sub(idx), self.Wg)
            alpha_i_sum += Di*z*z*ui_prev_g
            alpha_e_sum += De*z*z*ue_prev_g

            if dirichlet_bcs:
                # add Dirichlet boundary conditions on exterior boundary
                bc = DirichletBC(self.We.sub(idx), ion['ke_init'], self.exterior_boundary, 2)
                self.bcs.append(bc)

            # calculate and update Nernst potential for current ion
            ion['E'] = project(R*temperature/(F*z)*ln(ue_prev_g/ui_prev_g), self.Wg)

            # get ion channel current
            if self.MMS is None:
                ion['I_ch'] = ion['g_k']*(self.phi_M_prev - ion['E'])
            else:
                ion['I_ch'] = self.phi_M_prev
            # add contribution to total channel current
            I_ch += ion['I_ch']

        # Initialize the variational form
        a00 = 0; a01 = 0; a02 = 0; L0 = 0
        a10 = 0; a11 = 0; a12 = 0; L1 = 0
        a20 = 0; a21 = 0; a22 = 0; L2 = 0

        # Setup ion specific part of variational formulation
        for idx, ion in enumerate(self.ion_list):
            # get ion attributes
            z = ion['z']; Di = ion['Di']; De = ion['De']; I_ch_k = ion['I_ch']

            # Set intracellular ion attributes
            ki = ui[idx]           # unknown
            ki_prev = ui_prev[idx] # previous solution
            vki = vi[idx]          # test function
            # Set extracellular ion attributes
            ke = ue[idx]           # unknown
            ke_prev = ue_prev[idx] # previous solution
            vke = ve[idx]          # test function
            # Interpolate the previous solution on Gamma
            ki_prev_g = interpolate(ui_p.sub(idx), self.Wg)
            ke_prev_g = interpolate(ue_p.sub(idx), self.Wg)
            # Set fraction of ion specific intra--and extracellular I_cap
            alpha_i = Di*z*z*ki_prev_g/alpha_i_sum
            alpha_e = De*z*z*ke_prev_g/alpha_e_sum

            # linearised ion fluxes
            Ji = - Constant(Di)*grad(ki) - Constant(Di*z/psi)*ki_prev*grad(phi_i)
            Je = - Constant(De)*grad(ke) - Constant(De*z/psi)*ke_prev*grad(phi_e)

            # weak form - equation for k_i
            a00 += dt_inv*ki*vki*dxi - inner(Ji, grad(vki))*dxi
            a02 += 1.0/(F*z)*alpha_i*p_IM*vki*dxGamma
            L0  += dt_inv*ki_prev*vki*dxi \
                 - 1.0/(F*z)*(I_ch_k - alpha_i*I_ch)*vki*dxGamma

            # weak form - equation for k_e
            a11 += dt_inv*ke*vke*dxe - inner(Je, grad(vke))*dxe
            a12 -= 1.0/(F*z)*alpha_e*p_IM*vke*dxGamma
            L1  += dt_inv*ke_prev*vke*dxe \
                 + 1.0/(F*z)*(I_ch_k - alpha_e*I_ch)*vke*dxGamma

            # add contribution to total current flux
            J_phi_i += F*z*Ji
            J_phi_e += F*z*Je

            if self.MMS is not None:
                # MMS: add source terms
                L0 += inner(ion['f_k_i'], vki)*dxi # eq for k_i
                L1 += inner(ion['f_k_e'], vke)*dxe # eq for k_e
                # MMS: exterior boundary terms (zero in "physical" problem)
                L1 -= inner(dot(ion['J_k_e'], self.n_outer), vke)*dsOuter        # eq for k_e
                L1 += F*z*inner(dot(ion['J_k_e'], self.n_outer), vphi_e)*dsOuter # eq for phi_e

        # weak form - equation for phi_i
        a00 += inner(J_phi_i, grad(vphi_i))*dxi
        a02 -= inner(p_IM, vphi_i)*dxGamma

        # weak form - Lagrange terms (enforcing int phi_i = 0)
        a00 += _c*vphi_i*dxi + _d*phi_i*dxi

        # weak form - equation for phi_e
        a11 += inner(J_phi_e, grad(vphi_e))*dxe
        a12 += inner(p_IM, vphi_e)*dxGamma

        # weak form - equation for p_IM
        a20 += inner(phi_i, q_IM)*dxGamma
        a21 -= inner(phi_e, q_IM)*dxGamma
        a22 -= dt/C_M*inner(p_IM, q_IM)*dxGamma
        L2 += inner(self.phi_M_prev, q_IM)*dxGamma
        # add contribution of channel current to equation for phi_M
        if not splitting_scheme: L2 += - dt/C_M*inner(I_ch, q_IM)*dxGamma

        if self.MMS is not None:
            # MMS: add source terms
            L0 += inner(ion['phi_i_e'], _d)*dxi     # Lagrange for phi_i
            L0 += inner(ion['f_phi_i'], vphi_i)*dxi # equation for phi_i
            L1 += inner(ion['f_phi_e'], vphi_e)*dxe # equation for phi_e

            # coupling condition I_M = - J_e + g enforced in equation for phi_e
            L1 += sum(inner(gM, vphi_e)*dxGamma(i) for i, gM in\
                    enumerate(ion['f_g_M'], 1))
            # equation for phi_M (I_M in variational form)
            L2 += dt/C_M*sum(inner(JM, q_IM)*dxGamma(i) for i, JM in \
                    enumerate(ion['f_I_M'], 1))

        # Solution software does not support empty blocks -> hack
        a01 = Constant(0.0)*ke*vki*dxGamma
        a10 = Constant(0.0)*ki*vke*dxGamma

        # gather weak form in matrix structure
        self.a = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
        self.L = L0 + L1 + L2

        # create function for previous (known) solution
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
        self.matrix_blocks[5] = assemble_mixed(self.alist[5])
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

        comm = self.exterior_mesh.mpi_comm()
        w = Vector(comm, self.Wi.dim() + self.We.dim() + self.Wg.dim())

        solver = PETScLUSolver()        # create LU solver
        ksp = solver.ksp()              # get ksp  solver
        pc = ksp.getPC()                # get pc
        pc.setType("lu")                # set solver to LU
        pc.setFactorSolverType("mumps") # set LU solver to use mumps

        opts = PETSc.Options()          # get options
        opts["mat_mumps_icntl_4"] = 1   # set amount of info output
        opts["mat_mumps_icntl_14"] = 40 # set percentage of ???
        ksp.setFromOptions()            # update ksp with options set above

        solver.solve(AA, w, bb)             # solve system

        # Assign the obtained solution to function wh defined on the FunctionSpaceProduct
        w0 = Function(self.Wi).vector()
        w0.set_local(w.get_local()[:self.Wi.dim()])
        w0.apply('insert')
        self.wh.sub(0).assign(Function(self.Wi,w0))

        w1 = Function(self.We).vector()
        w1.set_local(w.get_local()[self.Wi.dim():self.Wi.dim()+self.We.dim()])
        w1.apply('insert')
        self.wh.sub(1).assign(Function(self.We,w1))

        w2 = Function(self.Wg).vector()
        w2.set_local(w.get_local()[self.Wi.dim()+self.We.dim():])
        w2.apply('insert')
        self.wh.sub(2).assign(Function(self.Wg,w2))

        # update previous ion concentrations
        self.u_p.sub(0).assign(self.wh.sub(0))
        self.u_p.sub(1).assign(self.wh.sub(1))
        self.u_p.sub(2).assign(self.wh.sub(2))
        # update previous membrane potential
        self.phi_M_prev.assign(interpolate(self.u_p.sub(0).sub(self.N_ions), self.Wg) \
                             - interpolate(self.u_p.sub(1).sub(self.N_ions), self.Wg))

        # updates problems time t
        self.t.assign(float(self.t + dt))

        # update Nernst potential for all ions
        for idx, ion in enumerate(self.ion_list):
            z = ion['z']
            ki_prev_g = interpolate(self.u_p.sub(0).sub(idx), self.Wg)
            ke_prev_g = interpolate(self.u_p.sub(1).sub(idx), self.Wg)
            ion['E'].assign(project(R*temperature/(F*z)*ln(ke_prev_g/ki_prev_g), self.Wg))
        return

    def solve_system_passive(self, filename, dirichlet_bcs=False):
        """ solve KNP-EMI with passive dynamics on membrane """

        # create variational formulation
        self.create_variational_form(dirichlet_bcs=dirichlet_bcs)

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
        self.initialize_xdmf_savefile(filename + '/')

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
        self.h5_file = HDF5File(self.interior_mesh.mpi_comm(), filename, 'w')
        self.h5_file.write(self.mesh, '/mesh')
        self.h5_file.write(self.gamma_mesh, '/gamma_mesh')
        self.h5_file.write(self.subdomains, '/subdomains')
        self.h5_file.write(self.surfaces, '/surfaces')

        self.h5_file.write(self.u_p.sub(1), '/exterior_solution',  self.h5_idx)
        self.h5_file.write(self.u_p.sub(0), '/interior_solution',  self.h5_idx)
        self.h5_file.write(self.phi_M_prev, '/membrane_potential', self.h5_idx)
        return

    def save_h5(self):
        """ save results to h5 file """
        self.h5_idx += 1
        self.h5_file.write(self.u_p.sub(0), '/interior_solution',  self.h5_idx)
        self.h5_file.write(self.u_p.sub(1), '/exterior_solution',  self.h5_idx)
        self.h5_file.write(self.phi_M_prev, '/membrane_potential', self.h5_idx)
        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()
        return

    def initialize_xdmf_savefile(self, file_prefix):
        """ initialize xdmf file """
        self.interior_xdmf_files = []
        self.exterior_xdmf_files = []
        ion_list_hack = self.ion_list + [{'name': 'phi'}]
        for idx, ion in enumerate(ion_list_hack):
            filename_xdmf = file_prefix + 'interior_' +  ion['name'] + '.xdmf'
            xdmf_file = XDMFFile(self.interior_mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters['rewrite_function_mesh'] = False
            xdmf_file.parameters['flush_output'] = True
            self.interior_xdmf_files.append(xdmf_file)
            xdmf_file.write(self.u_p.sub(0).split()[idx], self.t.values()[0])

            filename_xdmf = file_prefix + 'exterior_' +  ion['name'] + '.xdmf'
            xdmf_file = XDMFFile(self.exterior_mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters['rewrite_function_mesh'] = False
            xdmf_file.parameters['flush_output'] = True
            self.exterior_xdmf_files.append(xdmf_file)
            xdmf_file.write(self.u_p.sub(1).split()[idx], self.t.values()[0])

        filename_xdmf = file_prefix + 'membrane_potential' + '.xdmf'
        self.membrane_xdmf_file = XDMFFile(self.gamma_mesh.mpi_comm(), filename_xdmf)
        self.membrane_xdmf_file.parameters['rewrite_function_mesh'] = False
        self.membrane_xdmf_file.parameters['flush_output'] = True
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])
        return

    def save_xdmf(self):
        """ save results to xdmf files """
        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].write(self.u_p.sub(0).split()[i], self.t.values()[0])
            self.exterior_xdmf_files[i].write(self.u_p.sub(1).split()[i], self.t.values()[0])
        self.membrane_xdmf_file.write(self.phi_M_prev, self.t.values()[0])
        return

    def close_xdmf(self):
        """ close xdmf files """
        for i in range(len(self.interior_xdmf_files)):
            self.interior_xdmf_files[i].close()
            self.exterior_xdmf_files[i].close()
        self.membrane_xdmf_file.close()
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
