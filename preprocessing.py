import logging

# basic config must be done before loading other packages
# logger.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
logger = logging.getLogger(__file__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

import argparse
from pathlib import Path
import hyperspy.api as hs
import pyxem as pxm
from skimage.io import imsave
from pyxem.signals import LazyDiffraction2D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tabulate import tabulate
from skimage.feature import blob_dog, blob_log, blob_doh
from pathlib import Path

from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from skimage.morphology import erosion
from skimage.morphology import disk
import skimage.filters as skifi


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def subtract_background_dog(array, sigma_min, sigma_max):
    """
    Subtract the background of a 2D array using difference of gaussians (DOG)

    :param array: The image
    :param sigma_min: The lower sigma
    :param sigma_max: The upper sigma
    :return:
    """
    blur_max = gaussian_filter(array, sigma_max)
    blur_min = gaussian_filter(array, sigma_min)
    return np.maximum(np.where(blur_min > blur_max, array, 0) - blur_max, 0)


def preprocess(filename, lazy=True, com_mask=(128, 128, 12), formats=None):
    formats = formats if formats else ('.hspy', '.zarr')
    filename = Path(filename)
    logger.info(f'Loading data from {filename}')
    signal = hs.load(filename, lazy=lazy)
    logger.debug(f'Loaded data')
    if not isinstance(signal, pxm.signals.ElectronDiffraction2D):
        logger.warning(
            f'Only ElectronDiffraction2D signals can be proeprocessed. I got {signal!r} of type {type(signal)}')

    # Centering
    logger.info('Centering dataset')
    logger.debug(f'Calculating COM within {com_mask}')
    com = signal.center_of_mass(mask=com_mask)
    beam_shift = pxm.signals.BeamShift(com.T)
    mask = hs.signals.Signal2D(np.zeros(signal.axes_manager.navigation_shape, dtype=bool).T).T
    mask.inav[20:-20, 20:-20] = True
    logger.debug(f'Estimating linear plane')
    beam_shift.make_linear_plane(mask=mask)
    beam_shift = beam_shift - (signal.axes_manager.signal_shape[0] // 2)
    logger.info(
        f'Beam shifts are within {beam_shift.min(axis=[0, 1, 2, 3])} pixels and {beam_shift.max(axis=[0, 1, 2, 3])} pixels')
    signal.shift_diffraction(beam_shift.isig[0], beam_shift.isig[1], inplace=True)
    signal.metadata.add_dictionary({
        'Preprocessing': {
            'Centering': {
                'COM': com,
                'COM_mask': {
                    'x': com_mask[0],
                    'y': com_mask[1],
                    'r': com_mask[2]
                },
                'Shifts': beam_shift,
                'shift_estimate_mask': mask
            }
        }
    })

    # Calibration
    calibration = 0.009520  # Å^-1
    logger.info(f'Setting calibration to {calibration} Å^-1')
    signal.set_diffraction_calibration(calibration)

    # Binning
    binning = (1, 1, 2, 2)
    logger.info(f'Binning signal with scales {binning}')
    signal = signal.rebin(scale=binning)

    # Preparing masks
    logger.info(f'Preparing masks')
    image = signal.mean(axis=[0, 1])
    minimum_r = 5
    blob_kwargs = {
        'min_sigma': 1,
        'max_sigma': 15,
        'num_sigma': 100,
        'overlap': 0,
        'threshold': 1E-18,
    }
    sep = "\n\t"
    logger.info(
        f'Searching for blobs using arguments:\n\t{f"{sep}".join([f"{key}: {blob_kwargs[key]}" for key in blob_kwargs])}')
    blobs = blob_log(image.data, **blob_kwargs)
    nx, ny = image.axes_manager.signal_shape
    mask = np.zeros((nx, ny), dtype=bool)
    direct_beam_mask = np.zeros((nx, ny), dtype=bool)
    xs, ys = np.arange(0, nx), np.arange(0, ny)
    X, Y = np.meshgrid(xs, ys)
    for blob in blobs:
        y, x, r = blob  # x and y axes are flipped in hyperspy compared to numpy
        r = np.sqrt(2) * r  # Scale blob radius to appear more like a real radius
        r = max([minimum_r, r])  # Make sure that the radius is at least the specified minimum radius
        logger.info(f'Adding mask with radius {r} at ({x}, {y})')
        R = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

        # If the blob is within +/- 2 pixels of the center of the pattern, assign the blob to the direct beam mask
        if nx // 2 - 2 <= x <= nx // 2 + 2 and ny // 2 - 2 <= y <= ny // 2 + 2:
            logger.info(f'\tBlob assigned to direct beam mask')
            direct_beam_mask[R <= r] = True
        else:
            logger.info(f'\tBlob assigned to reflection mask')
            mask[R < r] = True

    signal.metadata.add_dictionary({
        'Preprocessing': {
            'Masks': {
                'Diffraction': {
                    'direct_beam': direct_beam_mask,
                    'reflections': mask,
                }
            }
        }
    })

    # Normalize. This makes the data a float type. Remember to change dtype back to uint 16 if needed!
    logger.info('Normalizing data')
    signal = signal / signal.nanmax(axis=[0, 1, 2, 3])

    # Rechunk to make sure chunking is reasonable
    nav_chunks = 32
    sig_chunks = 32
    logger.info(f'Rechunking with {nav_chunks} navigation chunks and {sig_chunks} signal chunks in each dimension')
    signal.rechunk(nav_chunks=nav_chunks, sig_chunks=sig_chunks)

    # Make VBF and maximum through-stack
    logger.info(f'Preparing VBF')
    vbf = signal.get_integrated_intensity(hs.roi.CircleROI(cx=0., cy=0., r_inner=0., r=0.07))
    signal.metadata.add_dictionary({
        'Preprocessing': {'VBF': vbf}
    })

    logger.info('Preparing maximum through-stack')
    maximums = signal.max(axis=[0, 1])
    signal.metadata.add_dictionary({
        'Preprocessing': {'Maximums': maximums}
    })

    # Save the signal
    if isinstance(formats, str):
        formats = [formats]

    for f in formats:
        preprocessed_filename = filename.with_name(f'{filename.stem}_preprocessed{f}')
        logger.info(f'Saving preprocessed data to "{preprocessed_filename.absolute()}"')
        try:
            signal.save(preprocessed_filename, chunks=(nav_chunks, nav_chunks, sig_chunks, sig_chunks), overwrite=True)
        except Exception as e:
            logger.error(f'Exception when saving preprocessed signal with format {f}: \n{e}. \nSkipping format and continuing.')


    # Save the VBF and maximums
    logger.info(f'Saving VBF and maximums as images')
    imsave(filename.with_name(f'{filename.stem}_preprocessed_vbf.png'), vbf.data)
    imsave(filename.with_name(f'{filename.stem}_preprocessed_maximums.png'), maximums.data)
    imsave(filename.with_name(f'{filename.stem}_preprocessed_maximums_log10.png'), np.log10(maximums.data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=Path, help='Path to a 4D-STEM dataset to convert')
    parser.add_argument('-v', '--verbose', dest='verbosity', default=0, action='count', help='Set verbose level')
    arguments = parser.parse_args()

    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min([arguments.verbosity, 2])]
    logger.setLevel(log_level)

    preprocess(arguments.filename, lazy=False, com_mask=(127, 126, 12.5), formats=('.hspy', '.zarr'))

