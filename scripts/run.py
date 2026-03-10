import argparse
import logging
import os
import pathlib
import tomllib

from ampy.ampy import AMPy
from ampy.products.minimize import main as run_minimizer


def parse_args():
    """ Optional command line arguments. """
    parser = argparse.ArgumentParser(description="AMPy Parameters")
    parser.add_argument('--obs', help='Observation path')
    parser.add_argument('--out', help='Output directory')
    parser.add_argument('--registry', help='Registry config path')
    parser.add_argument('--inf_cfg', help='Inference config path')
    return parser.parse_args()


def generate_corner_plots(ampy, output_dir):
    """
    Generate custom organized corner plots.

    Parameters
    ----------
    ampy : ``ampy.ampy.AMPy``
        The completed ampy instance.

    output_dir : ``pathlib.Path``
        The output directory.
    """
    model_corner_path = output_dir / "model_corner.pdf"
    dust_corner_path = output_dir / "dust_corner.pdf"
    stat_corner_path = output_dir / "stat_corner.pdf"

    ampy.corner_plot(('afterglow_flux',), model_corner_path)
    ampy.corner_plot(('source_frame_dust', 'milky_way_dust'), dust_corner_path)
    ampy.corner_plot(('calibration', 'chi_squared'), stat_corner_path)


def main(obs, registry, inf_cfg, output_dir,):
    """
    Main routine for running MCMC with AMPy.

    Parameters
    ----------
    obs : str or ``pathlib.Path``
        The observation filepath.

    registry : str or ``pathlib.Path``
        The path to the run config file.

    inf_cfg : str or ``pathlib.Path``
        The path to the inference config file.

    output_dir : str or ``pathlib.Path``
        The output directory.

    Returns
    -------
    int
    """

    # Load the inference parameters
    with open(inf_cfg, "rb") as f:
        inf_data = tomllib.load(f)

    # Create AMPy instance from run configuration
    ampy = AMPy.from_registry(obs, registry)

    # Run MCMC and get the most likely parameters
    logging.info("Starting the AMPy run...")

    best = ampy.run_mcmc(**inf_data)

    # Generate the following data products:
    #   - Light curve (brightness vs. time)
    #   - Spectral plot (spectral breaks and indices)
    #   - Density profile (density vs. blast radius)
    #   - Corner plot (posterior distributions)
    #   - Summary JSON report (best values, etc.)
    logging.info("Generating the data products...")

    ampy.generate_products(output_dir)

    # Generate the corner plots
    # The above call to ``generate_products`` already generates a corner plot,
    # but it is a mega-all-in-one-plot. Each user will have a preference on
    # how to split the corners, so do it the way we want here.
    generate_corner_plots(ampy, output_dir)

    # Run the minimizer using the MCMC best-fit as the starting point
    logging.info("Running the minimizer...")

    run_minimizer(
        obs_path=obs,
        registry_path=registry,
        report_path=output_dir / "report.json",
        results_dir=output_dir,
    )

    logging.info("Completed successfully...")

    return 0


if __name__ == "__main__":
    args = parse_args()

    output = pathlib.Path(args.out)

    if not os.path.exists(output):
        os.makedirs(output)

    main(args.obs, args.registry, args.inf_cfg, output)
