from typing import Dict, List, Optional, Tuple

from dolfin import Mesh, MeshFunction, MeshView, Measure, FacetNormal
from dolfin.cpp.mesh import MeshFunctionSizet


class SubDomainInfo:
    def __init__(
        self, name: str, value: int, interfaces: List[str], ext_bdrys: List[str]
    ):
        self.name = name
        self.value = value
        self.external_boundaries = ext_bdrys
        self.interfaces = interfaces


class FacetDomainInfo:
    def __init__(
        self, name: str, value: int, bordering_regions: Optional[Tuple[str, str]] = None
    ):
        self.name = name
        self.value = value
        self.bordering_regions = bordering_regions

    @property
    def external(self):
        return self.bordering_regions is None


class FacetView(Mesh):
    def __init__(self, boundaries: MeshFunctionSizet, info: FacetDomainInfo):
        super().__init__(MeshView.create(boundaries, info.value))
        self.rename(info.name, "")
        self.value = info.value
        self.bordering_regions = info.bordering_regions
        self.external = info.external

    def mark_facets(self, subdomain: Mesh, subdomainbdry: MeshFunctionSizet):
        """Label a meshfunction defined on subdomain with the value attributed to submesh."""
        self.build_mapping(subdomain)
        facetmap = self.topology().mapping()[subdomain.id()].cell_map()
        for facet in facetmap:
            subdomainbdry[facet] = self.value
        return facetmap

    def get_measure(self):
        return Measure("dx", domain=self)


class InterfaceMesh(FacetView):
    def __init__(self, boundaries: MeshFunctionSizet, info: FacetDomainInfo):
        super().__init__(boundaries, info)
        regions = info.bordering_regions
        self.inside = regions[0]
        self.outside = regions[1]

    def orientation(self, region):
        if region == self.inside:
            return 1.0
        elif region == self.outside:
            return -1.0
        return 0


class SubdomainView(Mesh):
    def __init__(self, subdomains: MeshFunctionSizet, subdomain_info: SubDomainInfo):
        super().__init__(MeshView.create(subdomains, subdomain_info.value))
        self.rename(subdomain_info.name, "")
        self.value = subdomain_info.value
        self.external_boundaries: List[FacetView] = []
        self.interfaces: List[FacetView] = []

        self.boundary_tags = MeshFunction("size_t", self, self.topology().dim() - 1, 0)
        self.n = FacetNormal(self)

    def mark_boundaries(
        self, external_bdrys: List[FacetView], interfaces: List[FacetView]
    ):
        self.external_boundaries = external_bdrys
        self.interfaces = interfaces
        for bdry in (*external_bdrys, *interfaces):
            bdry.mark_facets(self, self.boundary_tags)
        return self.boundary_tags

    def get_measure(self):
        return Measure("dx", domain=self)

    def get_external_boundary_measure(self):
        return Measure(
            "ds",
            domain=self,
            subdomain_data=self.boundary_tags,
            subdomain_id=tuple(bdry.value for bdry in self.external_boundaries),
        )  # TODO: Possible bug. Might need to be called after boundary marking.


class SubMeshCollection:
    def __init__(
        self,
        mesh: Mesh,
        cellfunction: MeshFunctionSizet,
        facetfunction: MeshFunctionSizet,
        subdomains_info: List[SubDomainInfo],
        boundaries_info: List[FacetDomainInfo],
    ):
        # Include mesh and subdomain information to adapt to file storage.
        self.mesh = mesh
        self.cellfunction = cellfunction
        self.facetfunction = facetfunction
        self.boundaries = [
            FacetView(facetfunction, bdry) for bdry in boundaries_info if bdry.external
        ]
        self.interfaces = [
            InterfaceMesh(facetfunction, iface)
            for iface in boundaries_info
            if not iface.external
        ]
        self.subdomains = [
            SubdomainView(cellfunction, subdomain) for subdomain in subdomains_info
        ]
        self._create_boundary_maps(subdomains_info)

    def _create_boundary_maps(self, subdomains_info: List[SubDomainInfo]):
        for idx, subdomain in enumerate(subdomains_info):
            self.subdomains[idx].mark_boundaries(
                [
                    bdry
                    for bdry in self.boundaries
                    if bdry.name() in subdomain.external_boundaries
                ],
                [
                    iface
                    for iface in self.interfaces
                    if iface.name() in subdomain.interfaces
                ],
            )


def create_submesh(mesh: Mesh, meshfunction: MeshFunctionSizet, labels: List[int]) -> Mesh:
    characteristic_meshfunction = MeshFunction("size_t", mesh, meshfunction.dim())
    for label in labels:
        for idx in meshfunction.where_equal(label):
            characteristic_meshfunction[idx] = 1
    return MeshView.create(characteristic_meshfunction, 1)


def tag_subdomain_facets(
    parentmesh: Mesh,
    subdomain: Mesh,
    facetfunction: MeshFunctionSizet,
    subdomain_boundary_tags: List[int],
) -> MeshFunctionSizet:
    facet_tags = MeshFunction("size_t", subdomain, subdomain.topology().dim() - 1)
    for bdry in subdomain_boundary_tags:
        iface = create_submesh(parentmesh, facetfunction, [bdry])
        mark_subdomain_facets(facet_tags, iface, subdomain, bdry)
    return facet_tags


def mark_subdomain_facets(tags: MeshFunctionSizet, interface: Mesh, subdomain: Mesh, value: int) -> MeshFunctionSizet:
    interface.build_mapping(subdomain)
    mapping = interface.topology().mapping()[subdomain.id()].cell_map()
    for facet in mapping:
        tags[facet] = value
    return tags


def tag_interface_subdomains(interface: Mesh, parent_facetfunction: MeshFunctionSizet) -> MeshFunctionSizet:
    interface_tags = MeshFunction("size_t", interface, interface.topology().dim())
    facetmap = (
        interface.topology().mapping()[parent_facetfunction.mesh().id()].cell_map()
    )
    for idx, facet in enumerate(facetmap):
        interface_tags[idx] = parent_facetfunction[facet]
    return interface_tags
