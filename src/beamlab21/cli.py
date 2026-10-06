#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Unified ``beamlab21`` command-line interface.

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

    fit_p = _add_run_parser(sub, "fit", "fit Gaussian + Zernike models to a beam slice")
    fit_p.add_argument(
        "--coord", choices=["auto", "cartesian", "polar"], default=None,
        help="coordinate system of the input data (overrides coord_type in the config)")

    compute_p = _add_run_parser(sub, "compute",
                                "compute a beam model from saved coefficients")
    compute_p.add_argument(
        "--coord", choices=["cartesian", "polar"], default=None,
        help="output coordinate system (overrides coord_type in the config)")

    fit_all_p = sub.add_parser("fit-all",
                               help="fit all frequency channels in a beam cube")
    fit_all_p.add_argument("config", help="path to the YAML config file")
    fit_all_p.add_argument("--base-dir", default=None,
                           help="base directory for config-relative paths")
    fit_all_p.add_argument("--coord", choices=["auto", "cartesian", "polar"], default=None,
                           help="coordinate system (overrides coord_type in the config)")
    fit_all_p.add_argument("--channels", type=float, nargs="+", default=None,
                           metavar="MHZ",
                           help="frequencies in MHz to fit (default: all channels)")
    fit_all_p.add_argument("--datafile", default=None,
                           help="path to beam cube (overrides datafile in the config)")

    cst_p = sub.add_parser("cst",
                           help="convert a CST farfield export and fit it")
    cst_p.add_argument("cst_file", help="path to the CST farfield .txt export")
    cst_p.add_argument("--freq", type=float, required=True,
                       help="frequency of this export in MHz (e.g. 400)")
    cst_p.add_argument("--column", choices=["copol", "e"], default="copol",
                       help="amplitude column to use: 'copol' (default) or 'e' (total field)")
    cst_p.add_argument("--size", type=int, default=1501,
                       help="output Cartesian grid resolution (default: 1501)")
    cst_p.add_argument("--xy-max", type=float, default=75.0,
                       help="half-width of the Cartesian window in degrees (default: 75)")
    cst_p.add_argument("--cube-output", default=None,
                       help="where to save the converted .npz beam cube "
                            "(default: temp file, deleted after fitting)")
    cst_p.add_argument("--config", default="configs/config_fit.yaml",
                       help="fit config file (default: configs/config_fit.yaml)")
    cst_p.add_argument("--base-dir", default=None,
                       help="base directory for config-relative paths")
    cst_p.add_argument("--skip-fit", action="store_true",
                       help="convert only; do not run the fit pipeline")

    cst_stack_p = sub.add_parser("cst-stack",
                                 help="stack multiple CST exports into one frequency cube")
    cst_stack_p.add_argument("cst_files", nargs="+",
                              help="CST farfield .txt files (one per frequency)")
    cst_stack_p.add_argument("--freqs", type=float, nargs="+", required=True,
                              metavar="MHZ",
                              help="frequency in MHz for each file (same order as files)")
    cst_stack_p.add_argument("--output", required=True,
                              help="path to save the stacked .npz beam cube")
    cst_stack_p.add_argument("--column", choices=["copol", "e"], default="copol")
    cst_stack_p.add_argument("--size", type=int, default=1501)
    cst_stack_p.add_argument("--xy-max", type=float, default=75.0)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "fit":
        from beamlab21 import fit
        fit.run(args.config, base_dir=args.base_dir, coord_type=args.coord)
    elif args.command == "fit-all":
        from beamlab21 import fit
        fit.run_all(args.config, channels=args.channels, base_dir=args.base_dir,
                    coord_type=args.coord, datafile=args.datafile)
    elif args.command == "compute":
        from beamlab21 import compute
        compute.run(args.config, base_dir=args.base_dir, coord_type=args.coord)
    elif args.command == "cst":
        from beamlab21 import cst
        cst.run(
            args.cst_file,
            freq_mhz=args.freq,
            config_path=args.config,
            base_dir=args.base_dir,
            column=args.column,
            size=args.size,
            xy_max=args.xy_max,
            cube_output=args.cube_output,
            skip_fit=args.skip_fit,
        )
    elif args.command == "cst-stack":
        from pathlib import Path as _Path

        import numpy as _np

        from beamlab21 import cst
        x, y, freq_arr, data = cst.stack(
            args.cst_files,
            freqs_mhz=args.freqs,
            column=args.column,
            size=args.size,
            xy_max=args.xy_max,
        )
        out = _Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(str(out), x=x, y=y, freq=freq_arr, data=data)
        print(f"Stacked cube saved → {out}")


if __name__ == "__main__":
    main()
