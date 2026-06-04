"""
generate_data.py — drop-in replacement for pyrft's generate_data()

Dependencies: numpy, scipy, nibabel, nilearn
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel
from nilearn.input_data import NiftiMasker


# ---------------------------------------------------------------------------
# Helpers (ported from github.com/sjdavenport/pyrft)
# ---------------------------------------------------------------------------

def _fwhm2sigma(fwhm: float) -> float:
    """Convert a FWHM value to a sigma for gaussian_filter."""
    return fwhm / np.sqrt(8 * np.log(2))


def _statnoise(masksize: tuple, nsubj: int, fwhm: float, truncation: int = 0) -> np.ndarray:
    """
    Generate stationary noise: white Gaussian noise smoothed with a Gaussian
    kernel (given FWHM), optionally truncated to remove edge effects.

    Returns an array of shape (*masksize, nsubj).
    """
    sigma = _fwhm2sigma(fwhm)

    if truncation == 0:
        # Matches pyrft behaviour when truncation=0 is explicitly passed
        fieldsize = masksize + (nsubj,)
        data = np.random.randn(*fieldsize)
        for n in range(nsubj):
            data[..., n] = gaussian_filter(data[..., n], sigma=sigma)
        return data

    # Automatic truncation (pyrft default)
    trunc = int(4 * np.ceil(fwhm))
    t_masksize = tuple(np.asarray(masksize) + 2 * trunc)
    fieldsize = t_masksize + (nsubj,)
    data = np.random.randn(*fieldsize)
    for n in range(nsubj):
        data[..., n] = gaussian_filter(data[..., n], sigma=sigma)

    ndim = len(masksize)
    if ndim == 2:
        data = data[
            (trunc + 1):(masksize[0] + trunc + 1),
            (trunc + 1):(masksize[1] + trunc + 1),
            :,
        ]
    elif ndim == 3:
        data = data[
            (trunc + 1):(masksize[0] + trunc + 1),
            (trunc + 1):(masksize[1] + trunc + 1),
            (trunc + 1):(masksize[2] + trunc),
            :,
        ]
    else:
        raise ValueError(f"masksize must be a 2D or 3D tuple, got: {masksize}")

    # Normalise to unit variance
    data = data / np.mean(np.std(data, axis=ndim, ddof=1))
    return data


def _random_signal_locations(
    field: np.ndarray,
    masksize: tuple,
    categ: np.ndarray,
    C: np.ndarray,
    pi0: float,
    scale: float = 1.0,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Add randomly located signal to field and return (field_with_signal, signal_array).

    Parameters
    ----------
    field      : array (*masksize, nsubj)
    masksize   : tuple of spatial dimensions
    categ      : array (nsubj,) with values 0 or 1 (two groups)
    C          : array (n_contrasts, n_params), contrast matrix
    pi0        : proportion of truly null voxels
    scale      : signal amplitude
    rng        : numpy RandomState / Generator (optional)

    Returns
    -------
    field      : array (*masksize, nsubj) — data with signal added
    signal     : array (*masksize, n_contrasts) — true signal map
    """
    if rng is None:
        rng = np.random.RandomState(101)

    n_contrasts = C.shape[0]
    nvox = int(np.prod(masksize))
    m = nvox * n_contrasts

    ntrue = int(np.round(pi0 * m))
    signal_entries = np.zeros(m)
    signal_entries[ntrue:] = 1.0

    signal = np.zeros(masksize + (n_contrasts,))

    if pi0 < 1.0:
        rng.shuffle(signal_entries)
        shuffled_signal = signal_entries
        spatial_signal2add = np.zeros(masksize)

        for j in range(n_contrasts):
            contrast_signal = shuffled_signal[j * nvox:(j + 1) * nvox]
            signal[..., j] = contrast_signal.reshape(masksize)
            spatial_signal2add += signal[..., j]
            # categ is 0 or 1; contrast C = [[0, 1]] targets group 1
            subjects_with_this_contrast = np.where(categ == (j + 1))[0]
            for k in subjects_with_this_contrast:
                field[..., k] += scale * spatial_signal2add

    return field, signal


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_data(
    dim: int,
    FWHM: float,
    pi0: float,
    scale: float = 0.5,
    nsubjects: int = 500,
):
    """
    Generate simulated fMRI data.

    Parameters
    ----------
    dim        : side length of the 3D volume (dim x dim x dim cube)
    FWHM       : full width at half maximum of the Gaussian kernel (in voxels)
    pi0        : proportion of null voxels (0 = all signal, 1 = all null)
    scale      : signal amplitude (default 0.5)
    nsubjects  : total number of subjects, split evenly into two groups

    Returns
    -------
    X           : array (nsubjects/2, n_voxels) — group1 minus group0 differences
    beta_true   : array (n_voxels,) — masked true signal
    nifti_masker: fitted NiftiMasker
    """
    nsubjects_ = int(nsubjects / 2)
    masksize = (dim, dim, dim)

    # 1. Stationary noise (mirrors pyrft call with truncation=0)
    field = _statnoise(masksize, nsubjects, FWHM, truncation=0)

    # 2. Categories: first half = 0, second half = 1
    categ = np.array([0] * nsubjects_ + [1] * nsubjects_)
    C = np.array([[0, 1]])

    # 3. Add signal
    field, sig = _random_signal_locations(field, masksize, categ, C, pi0=pi0, scale=scale)

    # 4. Differential image (group1 - group0)
    subjects_with_0s = np.where(categ == 0)[0]
    subjects_with_1s = np.where(categ == 1)[0]
    one_sample_image = field[..., subjects_with_1s] - field[..., subjects_with_0s]

    # 5. Convert to NIfTI and apply masking
    affine = np.eye(4)
    fmri_img = nibabel.Nifti1Image(dataobj=one_sample_image, affine=affine)
    sig_img  = nibabel.Nifti1Image(dataobj=sig, affine=affine)  # single contrast

    nifti_masker = NiftiMasker()
    X = nifti_masker.fit_transform(fmri_img)
    beta_true = nifti_masker.transform(sig_img)[0]

    return X, beta_true, nifti_masker
