from typing import Dict, Optional, Union

from dolfin import Constant

from neuralthreesome.parameters import get_default_parameters


class Ion:
    def __init__(self, name: str, valence: float, diffusion_coefficient: Constant,
        initial_concentrations: Dict[str, Constant]):
        self.name = name
        self.z = valence
        self.D = diffusion_coefficient
        self.k_init = initial_concentrations
        self.E = {}  
        self.I_ch = {}
        self.g_k = {}


def get_default_ion_list(param=None):
    if param is None:
        param = get_default_parameters()

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
    return [Na, K, Cl]
