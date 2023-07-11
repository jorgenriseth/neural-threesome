#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
from neuralthreesome.meshprocessing import geo2hdf

parser = ArgumentParser(
    description="""
                geo2dolfin
                --------------
                Convert a (2D) gmsh geo-file to a dolfin mesh with subdomains."""
)

parser.add_argument("infile", type=str)
parser.add_argument("outfile", type=str)
parser.add_argument("--cellsize", type=float, default=1.0)
parser.add_argument("--dim", type=int, default=2)
parser.add_argument(
    "--tmpdir",
    type=str,
    default="./tmp/",
    help='Defaults to "./tmp If provided, any intermediate files'
    "required in the creation of the mesh file will be saved in this directory.",
)
parser.add_argument("--keeptmp", action="store_true")
args = parser.parse_args()

# Parse paths, and create
infile = Path(args.infile)
meshfile = Path("{}".format(args.tmpdir)) / "{}".format(infile.with_suffix(".msh"))
outfile = Path(args.outfile)

geo2hdf(infile, outfile, dim=args.dim, tmpdir=args.tmpdir, cell_size=args.cellsize, cleanup=(not args.keeptmp))
