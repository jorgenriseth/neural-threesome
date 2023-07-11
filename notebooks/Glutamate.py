import numpy as np
from tqdm import tqdm
import dolfin as df
from pathlib import Path


class GlMesh:
    def __init__(self):
        self.mesh = None
        self.subdomain_fn = None
        self.boundary_fn = None
        self.domains = []
        self.boundaries = []

    def load_xdmf(
        self,
        mesh_dir: Path,
        subdom_dict: dict,
        bound_dict: dict,
    ) -> None:
        """
        Loads all mesh related data from a specific directory following the
        name scheme:
            1. Mesh: mesh.xdmf
            2. Subdomains: subdomains.xdmf
            3. Boundaries: boundaries.xdmf
        Number and tags of subdomains are expected to be specified and passed in
        subdom_dict and boundary_dict as follows:
        __dict = {
            'tag_1': associated_int_number_1,
            'tag_2': associated_int_number_2,
            ...
        }
        """
        # load mesh
        mesh_path = mesh_dir / Path("mesh.xdmf")
        self.mesh = df.Mesh()
        with df.XDMFFile(str(mesh_path.resolve())) as file:
            file.read(self.mesh)

        # load subdomain function
        subdomain_fn_path = mesh_dir / Path("subdomains.xdmf")
        self.subdomain_fn = df.MeshFunction(
            "size_t",
            self.mesh,
            self.mesh.topology().dim(),
            value=0,
        )
        with df.XDMFFile(str(subdomain_fn_path.resolve())) as file:
            file.read(self.subdomain_fn)

        # load boundary function
        boundary_coll_path = mesh_dir / Path("boundaries.xdmf")
        boundary_coll = df.MeshValueCollection(
            "size_t",
            self.mesh,
            self.mesh.topology().dim() - 1,
        )
        with df.XDMFFile(str(boundary_coll_path.resolve())) as file:
            file.read(boundary_coll)
        self.boundary_fn = df.MeshFunction(
            "size_t",
            self.mesh,
            boundary_coll,
        )

        # create subdomains
        for key in subdom_dict:
            sub_mesh = df.MeshView.create(self.subdomain_fn, subdom_dict[key])
            self.domains.append(sub_mesh)

        # create boundary subdomains
        for key in bound_dict:
            sub_mesh = df.MeshView.create(self.boundary_fn, bound_dict[key])
            self.boundaries.append(sub_mesh)

class Model():
    dims = {
        "dt": 0.0,
            }

class VesicleMSD(df.UserExpression):
    def __init__(self, model: Model, ti=[0.0], offsets=[0.0], **kwargs):
        super(VesicleMSD, self).__init__(**kwargs)
        self.dt = model.dims["dt"]
        self.cv = model.P["cv"]
        self.dv = model.P["dv"]
        self.wv = df.sqrt(df.pi) * self.dv / 2.0
        self.pS = 2.0 * self.cv * self.dv / (3.0 * self.dt)
        self.ti = ti
        self.offsets = np.array(offsets)

    def eval(self, value, x):
        atol = self.dt * 1e-2
        delta_tj_ti = [df.near(self.t, t, atol) for t in self.ti]
        if not any(delta_tj_ti):
            value[0] = 0.0
            return

        idx = delta_tj_ti.index(True)
        offset = self.offsets[idx]
        rho_on_ves = abs(x[1] - offset) <= (self.wv / 2.0)
        value[0] = rho_on_ves * self.pS

    def value_shape(self):
        return ()

class GlModel(Model):
    class State:
        def __init__(self, g: df.Function, m_AMPA: df.Function) -> None:
            self.g = g
            self.m_AMPA = m_AMPA

    # model specific parameter set
    pSet = {
        # diffusion coefficient in extra-cellular space
        "De": 3e5,  # nm^2 / ms
        # vesicular parameters
        "d_v": 40,  # nm
        "c_v": 60,  # mM
        # total uptake rate
        "k_tot": 10,  # ms^{-1}
        # geometry
        "w_cleft": 20,  # nm
        "w_astrocyte": 20,  # nm
        "h_domain": 400 / 2.0,  # nm
        "h_terminal": 360 / 2.0,  # nm
        # m_AMPA model parameters
        "alpha_AMPA": 1.1,  # (mM ms)^-1
        "beta_AMPA": 190e-3,  # ms^-1
    }

    # characteristic scales
    dims = {
        "c0": 1.0,  # mM
        "T": 1e-3,  # ms
        "W": pSet["w_cleft"],  # nm
        "H": pSet["h_domain"],  # nm
        # time step
        "dt": 1e-2,
    }

    # model specific subdomain key register
    subdom_dict = {
        "extra-cellular space": 1,
    }

    # model specific boundary subdomain key register
    boundary_dict = {
        "extra-cellular border": 1,
        "pre-synaptic terminal": 2,
        "post-synaptic terminal": 3,
        "astrocyte": 4,
        "dendritic membrane": 5,
    }

    def __init__(
        self,
        gl_mesh: GlMesh,
        params: dict = pSet,
        dimensions: dict = dims,
    ) -> None:
        self.mesh_obj = gl_mesh
        self.pSet = params
        self.dims = dimensions
        p = self.pSet
        d = self.dims

        # binding rate surface density
        k_bind = (p["k_tot"] / 5.0) / (2.0 * p["h_terminal"])
        k_uptake = (p["k_tot"] * 4.0 / 5.0) / (2.0 * p["w_cleft"])

        # define dimensionless parameter set
        self.P = {
            # diffusion tensor components
            "Dx": p["De"] * d["T"] / d["W"] ** 2.0,
            "Dy": p["De"] * d["T"] / d["H"] ** 2.0,
            # vesicular properties
            "cv": p["c_v"] / d["c0"],
            "dv": p["d_v"] / d["H"],
            # binding and uptake densities
            "k_bind": k_bind * d["T"] * d["H"] / d["W"],
            "k_uptake": k_uptake * d["T"] * d["W"] / d["H"],
            # m_AMPA model parameters
            "alpha_AMPA": p["alpha_AMPA"] * d["c0"] * d["T"],
            "beta_AMPA": p["beta_AMPA"] * d["T"],
        }
        # create function spaces
        idx_dom_e = [
            idx
            for idx, e in enumerate(list(GlModel.subdom_dict.keys()))
            if e == "extra-cellular space"
        ][0]
        self.V = df.FunctionSpace(self.mesh_obj.domains[idx_dom_e], "CG", 1)
        idx_dom_postsyn = [
            idx
            for idx, e in enumerate(list(GlModel.boundary_dict.keys()))
            if e == "post-synaptic terminal"
        ][0]
        self.Vpostsyn = df.FunctionSpace(
            self.mesh_obj.boundaries[idx_dom_postsyn], "DG", 2
        )

        # create empty state
        self.g0 = None
        self.m_AMPA0 = None

        # create empty variational forms
        self.L = None
        self.a = None

        # create empty solutions
        self.g = None
        self.m_AMPA = None

    def create_variational_problem(
        self,
        stimulus: VesicleMSD,
        s_init: State,
        dt: float,
    ) -> None:
        self.dims["dt"] = dt

        self.g0 = s_init.g
        self.m_AMPA0 = s_init.m_AMPA

        self.pS_g = stimulus

        # GLUTAMATE MODEL

        # create test and trial functions
        g = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)

        mesh = self.mesh_obj.mesh
        dx = df.Measure("dx", domain=mesh, subdomain_data=self.mesh_obj.subdomain_fn)
        ds = df.Measure("ds", domain=mesh, subdomain_data=self.mesh_obj.boundary_fn)

        # define variational problem
        n = df.FacetNormal(mesh)
        dt = self.dims["dt"]
        # rho = Expression('abs(x[1])', degree=2)
        D = df.as_matrix([[self.P["Dx"], 0.0], [0.0, self.P["Dy"]]])
        self.a = v * g * dx + dt * df.inner(df.grad(v), D * df.grad(g)) * dx
        self.L = (
            v * self.g0 * dx
            + dt
            * v
            * df.inner(D * df.grad(self.g0), n)
            * ds(self.boundary_dict["extra-cellular border"])
            + dt * v * self.pS_g * ds(self.boundary_dict["pre-synaptic terminal"])
            - dt
            * v
            * self.P["k_bind"]
            * self.g0
            * ds(self.boundary_dict["post-synaptic terminal"])
            - dt
            * v
            * self.P["k_uptake"]
            * self.g0
            * ds(self.boundary_dict["astrocyte"])
        )

        # AMPA model
        # define right hand side of the ODE
        def f_AMPA(m_AMPA, g):
            g_gamma_post = df.project(g, self.Vpostsyn)
            alpha = self.P["alpha_AMPA"]
            beta = self.P["beta_AMPA"]
            return alpha * g_gamma_post * (1.0 - m_AMPA) - beta * m_AMPA

        self.f_AMPA = f_AMPA

    def init(self, t: float, s0: State) -> None:
        self.pS_g.t = t
        self.g0 = s0.g
        self.m_AMPA0 = s0.m_AMPA

        self.g = df.Function(self.V)
        self.m_AMPA = df.Function(self.Vpostsyn)

    def expl_euler_step(self, t: float, s0: State) -> State:
        # initialize
        self.pS_g.t = t
        self.g0 = s0.g
        self.m_AMPA0 = s0.m_AMPA

        NoneType = type(None)
        if (type(self.m_AMPA) is NoneType) or ((self.g) is NoneType):
            raise ValueError(
                "Solutions are initialized as empty. Please initialize the model before solving."
            )

        # solve variational problem
        df.solve(self.a == self.L, self.g)
        self.g0.assign(self.g)

        # solve ODE problem point-wise
        self.m_AMPA = df.project(
            self.m_AMPA0
            + self.dims["dt"]
            * self.f_AMPA(
                self.m_AMPA0,
                self.g0,
            ),
            self.Vpostsyn,
        )
        
        self.m_AMPA0.assign(self.m_AMPA)

        return GlModel.State(self.g0, self.m_AMPA0)


class Solver:
    def __init__(self, model: GlModel):
        self.model = model
        self.out_path = None

    def solve(
        self,
        init: GlModel.State,
        stimulus: df.UserExpression,
        ts: tuple = (0, 20),
        dt=1e-2,
    ) -> GlModel.State:
        """
        Solve GlModel using an explicit Euler time discretization
        Args:
            init (GlModel.State) - initial state of the model containing g0 and m_AMPA0
            ts (tuple) - t0 and T of the interval [t0, T] to solve model on
            dt (float) - time step

        Vals:
            solution (GlModel.State) - solution at t=T
        """

        self.model.create_variational_problem(stimulus, init, dt)

        t = ts[0]
        t_fin = ts[1]
        num_steps = int(t_fin // dt) + 1
        self.model.init(t, init)

        solution = init

        if type(self.out_path) is not type(None):
            glutamate_vtk = df.File(str(self.out_path / Path("glutamate.pvd")))
            m_AMPA_vtk = df.File(str(self.out_path / Path("m_AMPA.pvd")))
            glutamate_vtk << (solution.g, t)
            m_AMPA_vtk << (solution.m_AMPA, t)

        for i_t in tqdm(range(num_steps)):

            t += dt
            solution = self.model.expl_euler_step(t, solution)

            if type(self.out_path) is not type(None):
                glutamate_vtk << (solution.g, t)
                m_AMPA_vtk << (solution.m_AMPA, t)

        return solution

if __name__ == "__main__":
    subdom_dict = GlModel.subdom_dict
    boundary_dict = GlModel.boundary_dict
    mesh_path = Path("./synapse_mesh/")
    gl_mesh = GlMesh()
    gl_mesh.load_xdmf(mesh_path, subdom_dict, boundary_dict)

    model = GlModel(gl_mesh)

    solver = Solver(model)
    solver.out_path = Path("./solution/")
    
    stimulus = VesicleMSD(model, ti=[0.1, 0.4, 10.], offsets=[0.0, -0.8, +0.8])
    
    # define initial state
    g0 = df.interpolate(df.Expression('0.0', degree=1), model.V)
    m_AMPA0 = df.interpolate(df.Expression('0.0', degree=1), model.Vpostsyn)
    init_state = GlModel.State(g0, m_AMPA0)
    
    # solve on time interval
    ts = (0., 20.)
    dt = 1e-2
    solution = solver.solve(init_state, stimulus, ts=ts, dt=dt)
