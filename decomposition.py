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
import pyxem as pxm
import numpy as np
import argparse
import time

from pathlib import Path


def decompose(
        signal,
        normalize_poissonian_noise=False,
        algorithm="SVD",
        output_dimension=None,
        navmask=None,
        diffmask=None,
        **kwargs
):
    if not isinstance(signal, hs.signals.Signal2D):
        raise TypeError(
            f"Cannot perform decomposition on {signal!r}. It is not a 2D hyperspy signal"
        )

    if navmask is not None:
        logger.debug(f"Got navigation mask {navmask}")
        if isinstance(navmask, hs.signals.Signal2D):
            logger.debug(f"Navigation mask is a hyperspy signal. Extracting data array and transposing the data")
            navmask = navmask.data.transpose() #The data needs to be transposed due to different conventions in hyperspy and numpy.
        if not signal.axes_manager.navigation_shape == navmask.shape:
            logger.warning(
                f"The navigation mask shape {navmask.shape} does not match signal navigation shape {signal.axes_manager.navigation_shape}"
            )

    if diffmask is not None:
        logger.debug(f"Got diffraction mask with {np.count_nonzero(diffmask)} nonzero values")
        if isinstance(diffmask, hs.signals.Signal2D):
            logger.debug(
                f"Diffraction mask is a hyperspy signal. Extracting data array"
            )
            diffmask = diffmask.data
        if not signal.axes_manager.signal_shape == diffmask.shape:
            logger.warning(
                f"The diffraction mask shape {diffmask.shape} does not match signal diffraction shape {signal.axes_manager.signal_shape}"
            )

    if (
            isinstance(signal, pxm.signals.LazyDiffraction2D)
            and arguments.algorithm == "NMF"
    ):
        logger.warning(
            f"Signal {signal} is lazy but specified algorithm {arguments.algorithm} is not compatible with lazy signals."
        )
        logger.warning(
            f"I will compute the signal to make it non-lazy and compatible with requested algorithm"
        )
        signal.compute()

    logger.info(f"Starting {algorithm} decomposition into {output_dimension} components with keyword arguments {kwargs}")
    tic = time.time()
    decomp = signal.decomposition(
        normalize_poissonian_noise=normalize_poissonian_noise,
        algorithm=algorithm,
        output_dimension=output_dimension,
        navigation_mask=navmask,
        signal_mask=diffmask,
        return_info=True,
        **kwargs,
    )
    toc = time.time()
    logger.info(f"Finished decomposition. Elapsed time: {toc - tic} seconds")
    logger.info(f"Decoposition parameters: {decomp}")

    if algorithm == 'NMF':
        logger.info(f"Decomposition reconstruction error: {decomp.reconstruction_err_}")
        logger.info(f"Decomposition number of iterations: {decomp.n_iter_}")

    return decomp


if __name__ == "__main__":
    # Parser arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hs_file", type=Path, help="The HyperSpy .hdf5 file to decompose"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Value to scale the diffraction patters with"
    )
    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Whether to apply a log10 scale to the data or not"
    )
    parser.add_argument(
        "--log_offset",
        type=float,
        help="Value to add to the data before logscaling it, and subsequently subtract the logarithm of this value from the logscaled data"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Whether to mask the data or not, using masks in the metadata. The masks should be True where data is to be removed/masked away.",
    )
    parser.add_argument(
        "--navmask",
        action="append",
        help="Append a navigation mask to use for masking the decomposition in real space. Specify multiple times to add more masks, or specify a single time with 'all' to select all navigation masks in the metadata field."
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        help="Whether to apply diffraction mask to the data before decomposition instead of supplying them to the decomposition algorithm.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Diffraction pattern cutoff in Å^-1"
    )
    parser.add_argument(
        "--algorithm",
        default="NMF",
        choices=["SVD", "NMF"],
        help="The decomposition algorithm to use",
    )
    parser.add_argument(
        "--components", default=None, type=int, nargs='+',
        help="Output components. Can be a series of components to decompose dataset into."
    )
    parser.add_argument(
        "--poissonian",
        action="store_true",
        help="Whether to assume poissonian noise when decomposing data or not",
    )
    parser.add_argument(
        "--max_iter",
        default=200,
        type=int,
        help='The number of maximum iterations used during NMF'
    )
    parser.add_argument(
        "--initialization",
        default=None,
        type=str,
        choices=[None, "random", "nndsvd", "nndsvda", "nndsvdar"],
        help="The initialization used for NMF"
    )
    parser.add_argument(
        "--random_state",
        default=None,
        type=int,
        help="The random state used when --initialization is random"
    )
    parser.add_argument(
        "--nocopy",
        action='store_false',
        help="Whether to not backup the data before the decomposition"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help=r"The path to store output. Default is the same directory as the input file",
    )
    parser.add_argument(
        "--store_signal",
        action="store_true",
        help="Whether to also store the decomposed signal with decomposition results or not. The factors and loadings will not be stored separately."
    )
    parser.add_argument("--lazy", action="store_true", help="Load lazy or not")
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float64"],
        help="The precision to use when performing NMF",
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
    logger.debug(f'Running decomposition script with arguments:{"".join(args_as_str)}')

    kwargs = {
        'random_state': arguments.random_state,
        'copy': arguments.nocopy,
    }

    if arguments.algorithm == "NMF":
        kwargs.update(
            {
                'max_iter': arguments.max_iter,
                'init': arguments.initialization
            }
        )

    if arguments.output_path is None:
        output_path = arguments.hs_file.parent
        logger.info(
            f'No output directory specified, I will put outputs at "{output_path.absolute()}"'
        )
    else:
        output_path = arguments.output_path
        if not output_path.exists():
            logger.info(
                f'Specified output path "{output_path.absolute()}" does not exists. I will attempt to create it'
            )
            logger.info(f'Creating output directory "{output_path.absolute()}"')
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory successfully.")
        logger.info(f'I will put outputs at "{output_path.absolute()}"')

    # Suffix to add to output name
    suffix = ""
    if False:
        if arguments.scale is not None:
            suffix += f"_scaled-{arguments.scale}"
        if arguments.poissonian:
            suffix += "_poissonian"
        if arguments.mask is not None:
            suffix += "_mask"
        if arguments.logscale:
            suffix += "_log"
            if arguments.log_offset is not None:
                suffix += f"_logoffset{arguments.log_offset}"
        for arg in kwargs:
            suffix += f"_{arg}-{kwargs[arg]}"

    logger.info(f'Loading data signal "{arguments.hs_file.absolute()}')
    signal = hs.load(arguments.hs_file.absolute(), lazy=arguments.lazy)

    if arguments.scale is not None:
        logger.info(f"Scaling signal by multiplying it with {arguments.scale}")
        signal = signal * arguments.scale

    # Check type and convert to float if needed
    if signal.data.dtype != arguments.precision:
        logger.info(
            f"Changing datatype from {signal.data.dtype} to {arguments.precision}"
        )
        signal.change_dtype(arguments.precision)

    if arguments.logscale:
        logger.info(f"Logscaling the data (base 10)")
        if arguments.log_offset is None:
            logger.info(f"Using no log offset")
            signal = np.log10(signal)
        else:
            logger.info(f"Using log offset {arguments.log_offset}")
            signal = np.log10(signal + arguments.log_offset, dtype=arguments.precision) - np.log10(arguments.log_offset,
                                                                                                   dtype=arguments.precision)

    if arguments.mask:
        # Extract diffraction mask
        logger.info(
            f"Getting diffraction mask signals"
        )
        try:
            masks = [
                mask[1] for mask in signal.metadata.Preprocessing.Masks.Diffraction
            ]  # Extract the diffraction masks from the metadata
        except AttributeError as e:
            logger.error(f"No Diffraction mask detected:\n{e}")
            diffmask = None
        else:
            logger.info(f"Found {len(masks)} diffraction masks in the metadata")
            diffmask = np.zeros(
                signal.axes_manager.signal_shape, dtype=bool
            )  # Create mask
            for mask in masks:  # Iterate and add masks together
                logger.info(f"Adding mask {mask} to diffraction mask")
                diffmask += mask

        # Extract navigation mask signals
        logger.info(f"Getting navigation mask signals")
        navmasks = arguments.navmask
        if navmasks is None:
            navmasks = []
        try:
            masks = signal.metadata.Preprocessing.Masks.Navigation  # Extract the navigation masks from the metadata
        except AttributeError as e:
            logger.error(f"No Navigation mask detected:\n{e}")
            navmask = None
        else:
            logger.info(f"Found {len(masks)} navigation masks in the metadata")
            if len(navmasks) == 0:
                logger.info("No navigation mask selection specified, skipping all navigation masks")
                navmask = None
            else:
                logger.info(f"Filtering masks based on selection: {navmasks}.")
                navmask = np.zeros(
                    signal.axes_manager.navigation_shape, dtype=bool
                )  # Create mask
                if len(navmasks) == 1 and navmasks[0] == 'all':
                    logger.info(f"Using all navigation masks present in metadata")
                    navmasks = list(masks.keys())
                for mask_name, mask in masks:  # Iterate and add masks together
                    if mask_name in navmasks:
                        logger.info(f"Adding mask {mask} to navigation mask")
                        navmask += mask
                        suffix += f"_navmask-{mask_name}"
                    else:
                        logger.info(
                            f"Did not add mask {mask} to navigation mask as the name '{mask_name}' was not found in navigation mask selection list {navmasks}")
                logger.debug(f"Transposing navigation mask")
                navmask = navmask.transpose()

        if arguments.apply_mask:
            logger.info(f"Applying diffraction masks to the signal")
            signal = signal * ~diffmask  # apply the diffraction mask to the signal
            diffmask = None  # Set the diffraction mask to None as the signal is already masked now.
    else:
        diffmask = None
        navmask = None

    # Cutoff signal
    if arguments.cutoff is not None:
        cutoff = abs(arguments.cutoff)
        logger.info(f"Cutting signal off at {cutoff} {signal.axes_manager[-1].units}")
        signal = signal.isig[-cutoff:cutoff + signal.axes_manager[-2].scale,
                 -cutoff:cutoff + signal.axes_manager[-1].scale]

    # Recheck type and convert if needed
    if signal.data.dtype != arguments.precision:
        logger.info(
            f"Changing datatype from {signal.data.dtype} to {arguments.precision}"
        )
        signal.change_dtype(arguments.precision)

    if arguments.components is None:
        components = [None]
    else:
        components = arguments.components

    for i, component in enumerate(components):
        logger.info(f"Running decomposition {i+1} of {len(components)}")
        decomp = decompose(
            signal,
            normalize_poissonian_noise=arguments.poissonian,
            algorithm=arguments.algorithm,
            output_dimension=component,
            navmask=navmask,
            diffmask=diffmask,
            **kwargs
        )

        # File saving

        output_name = (
                output_path
                / f"{arguments.hs_file.stem}_{arguments.algorithm}_{component}{suffix}{arguments.hs_file.suffix}"
        )
        logger.info(f'I will output data to "{output_name.absolute()}"')

        if arguments.store_signal:
            logger.info(f'Saving dataset with decomposition results to "{output_name}"')
            if decomp is not None:
                signal.metadata.add_dictionary({'Decomposition': decomp.__dict__})
            signal.save(output_name, overwrite=True)
        else:
            logger.info(
                f'Saving learning results to "{output_name}" (with _loadings and _factors name identifiers)'
            )

            try:
                logger.info(f"Saving factors")
                factors = signal.get_decomposition_factors()
                if decomp is not None:
                    factors.metadata.add_dictionary({'Decomposition': decomp.__dict__})
                factors.save(
                    output_name.with_name(f"{output_name.stem}_factors{output_name.suffix}"),
                    overwrite=True,
                )
            except Exception as e:
                logger.error(f"Could not save decomposition factors:\n{e}")

            try:
                logger.info(f"Saving loadings")
                loadings = signal.get_decomposition_loadings()
                if decomp is not None:
                    loadings.metadata.add_dictionary({'Decomposition': decomp.__dict__})
                loadings.save(
                    output_name.with_name(f"{output_name.stem}_loadings{output_name.suffix}"),
                    overwrite=True,
                )
            except Exception as e:
                logger.error("Could not save decomposition loadings: \n{e}")
        logger.info(f"Finshed decomposition {i+1} of {len(components)}\n")

        logger.info(f"Undoing treatments performed by the decomposition")
        signal.undo_treatments()
        logger.info(f"Undid treatments performed by the decomposition")

    logger.info(f"Finished decomposition script")
