import logging

logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

import hyperspy.api as hs
import argparse

from pathlib import Path

if __name__ == "__main__":
    # Parser arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hs_file", type=Path, help="The HyperSpy .hdf5 file to decompose"
    )

    parser.add_argument(
        "-l", "--lazy", action="store_true", help="Load data lazily or not"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        default=0,
        action="count",
        help="Set verbose level",
    )

    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][
        min([arguments.verbosity, 2])
    ]
    logger.setLevel(log_level)

    args_as_str = [
        f"\n\t{arg!r} = {getattr(arguments, arg)}" for arg in vars(arguments)
    ]

    logger.info(f'Loading data "{arguments.hs_file.absolute()}"')
    signal = hs.load(arguments.hs_file.absolute(), lazy=arguments.lazy)

    logger.info(f"Signal has metadata:\n{signal.metadata}")

    try:
        logger.info(f"Removing navigation masks")
        del signal.metadata.Preprocessing.Masks.Navigation
    except AttributeError as e:
        logger.error(f"Could not remove navigation masks due to error:\n{e}")
    else:
        logger.info(f"Signal now has metadata:\n{signal.metadata}")
        logger.info(f"Saving signal")
        signal.save(arguments.hs_file.absolute(), overwrite=True)
        logger.info(f"Saved file successfully")

    logger.info(f"Finished navigation mask cleanup script")
