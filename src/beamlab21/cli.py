#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Unified ``beamlab21`` command-line interface.

    beamlab21 fetch-data [--url URL] [--dest PATH] [--force]
    beamlab21 fit      CONFIG [--base-dir DIR]
    beamlab21 compute  CONFIG [--base-dir DIR]

The standalone ``beamlab21-fit`` / ``beamlab21-compute`` entry points remain
available and are equivalent to the ``fit`` / ``compute`` subcommands.
"""

import argparse

from beamlab21 import __version__


def _add_run_parser(sub, name, help_text):
    p = sub.add_parser(name, help=help_text)
    p.add_argument("config", help="path to the YAML config file")
    p.add_argument("--base-dir", default=None,
                   help="directory that relative paths in the config resolve against "
                        "(default: the config's project dir, else the current dir)")
    return p


def build_parser():
    parser = argparse.ArgumentParser(prog="beamlab21", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"beamlab21 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_run_parser(sub, "fit", "fit Gaussian + Zernike models to a beam slice")
    _add_run_parser(sub, "compute", "compute a beam model from saved coefficients")

    fd = sub.add_parser("fetch-data", help="download the example beam cube")
    fd.add_argument("--url", default=None, help="download URL (overrides env / DATA_URL.txt)")
    fd.add_argument("--dest", default=None, help="output path (default: data/Example_cube.npz)")
    fd.add_argument("--force", action="store_true", help="re-download even if the file exists")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "fit":
        from beamlab21 import fit
        fit.run(args.config, base_dir=args.base_dir)
    elif args.command == "compute":
        from beamlab21 import compute
        compute.run(args.config, base_dir=args.base_dir)
    elif args.command == "fetch-data":
        from beamlab21.data import fetch_example_data
        fetch_example_data(dest=args.dest, url=args.url, force=args.force)


if __name__ == "__main__":
    main()
