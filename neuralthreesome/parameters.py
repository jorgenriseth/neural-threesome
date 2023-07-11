from dolfin import Constant

# PARAMETERS = dict(
#     C_M = 0.02,                       # capacitance (F)
#     temperature = 300,                # temperature (K)
#     F = 96485,                        # Faraday's constant (C/mol)
#     R = 8.314,                        # Gas constant (J/(K*mol))
#     g_Na_leak = Constant(30*0.2 / (1e9)**2),     # Na leak membrane conductivity (S/(nm^2))
#     g_K_leak = Constant(30*0.8 / (1e9)**2),      # K leak membrane conductivity (S/(nm^2))
#     g_Cl_leak = Constant(0.0 / (1e9)**2),        # Cl leak membrane conductivity (S/(nm^2))
#     g_syn_bar = 1.25e3 / (1e9)**2,               # Na synaptic membrane conductivity (S/(m^2))
#     D_Na = Constant(1.33e-9 * (1e9)**2 / (1e3)),         # Na diffusion coefficient (nm^2/ms)
#     D_K = Constant(1.96e-9 * (1e9)**2 / (1e3)),          # K diffusion coefficient (nm^2/ms)
#     D_Cl = Constant(2.03e-9 * (1e9)**2 / (1e3)),         # Cl diffusion coefficient (nm^2/ms)
    
#     # EMI specific parameters
#     sigma_i = 2.01202,                # intracellular conductivity
#     sigma_e = 1.31365,                # extracellular conductivity
#     sigma_i_low = 1.0,                 # lower intracellular conductivities (from Bokil et al)
#     sigma_e_low = 0.05,                # lower extracellular conductivity
#     E_Na = 54.8e0,                   # reversal potential Na (mV)
#     E_K = -88.9e0,                  # reversal potential K (mV)


#     # TODO: Verify that using mMolar does not affect solution (as they are given in ( mmol / (dm^3) )
#     # initial conditions
#     phi_M_init = Constant(-60e0),    # membrane potential (mV)
#     Na_i_init = Constant(12),         # intracellular Na concentration (mol/m^3 = mmol / L) 
#     Na_e_init = Constant(100),        # extracellular Na concentration (mol/m^3 = mmol / L)
#     K_i_init = Constant(125),         # intracellular K concentration (mol/m^3 = mmol / L)
#     K_e_init = Constant(4),           # extracellular K concentration (mol/m^3 = mmol / L)
#     Cl_i_init = Constant(137),        # intracellular Cl concentration (mol/m^3 = mmol / L)
#     Cl_e_init = Constant(104),        # extracellular Cl concentration (mol/m^3 = mmol / L)
# )


PARAMETERS = dict(
    C_M = 0.02,                       # capacitance (F)
    temperature = 300,                # temperature (K)
    F = 96485,                        # Faraday's constant (C/mol)
    R = 8.314,                        # Gas constant (J/(K*mol))
    g_Na_leak = Constant(30*0.2),     # Na leak membrane conductivity (S/(m^2))
    g_K_leak = Constant(30*0.8),      # K leak membrane conductivity (S/(m^2))
    g_Cl_leak = Constant(0.0),        # Cl leak membrane conductivity (S/(m^2))
    g_syn_bar = 1.25e3,               # Na synaptic membrane conductivity (S/(m^2))
    D_Na = Constant(1.33e-9),         # Na diffusion coefficient (m^2/s)
    D_K = Constant(1.96e-9),          # K diffusion coefficient (m^2/s)
    D_Cl = Constant(2.03e-9),         # Cl diffusion coefficient (m^2/s)
    
    # EMI specific parameters
    sigma_i = 2.01202,                # intracellular conductivity
    sigma_e = 1.31365,                # extracellular conductivity
    sigma_i_low = 1.0,                 # lower intracellular conductivities (from Bokil et al)
    sigma_e_low = 0.05,                # lower extracellular conductivity
    E_Na = 54.8e-3,                   # reversal potential Na (V)
    E_K = -88.98e-3,                  # reversal potential K (V)


    # initial conditions
    phi_M_init = Constant(-60e-3),    # membrane potential (V)
    Na_i_init = Constant(12),         # intracellular Na concentration (mol/m^3 = mmol / L) 
    Na_e_init = Constant(100),        # extracellular Na concentration (mol/m^3 = mmol / L)
    K_i_init = Constant(125),         # intracellular K concentration (mol/m^3 = mmol / L)
    K_e_init = Constant(4),           # extracellular K concentration (mol/m^3 = mmol / L)
    Cl_i_init = Constant(137),        # intracellular Cl concentration (mol/m^3 = mmol / L)
    Cl_e_init = Constant(104),        # extracellular Cl concentration (mol/m^3 = mmol / L)
    n_init = 0.27622914792,                  # gating variable n
    m_init = 0.0379183462722,                # gating variable m
    h_init = 0.688489218108,                 # gating variable h
)

def get_default_parameters():
    return {**PARAMETERS}


