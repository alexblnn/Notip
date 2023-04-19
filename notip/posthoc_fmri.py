"""This package includes tweaked Nilearn functions (orig. author = B.Thirion)
and utilitary functions to use SansSouci on fMRI data (author = A.Blain)

"""
import warnings

import numpy as np
from scipy.stats import norm

from nilearn.input_data import NiftiMasker
from nilearn.glm import fdr_threshold
from nilearn.image import get_data, math_img, new_img_like

from nilearn.datasets import get_data_dirs
from scipy import stats
import sanssouci as sa
import os
import json
import pandas as pd
from tqdm import tqdm

from string import ascii_lowercase
from scipy import ndimage

from nilearn.image import threshold_img
from nilearn.image.resampling import coord_transform
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import _safe_get_data

from nilearn.reporting._get_clusters_table import _local_max


def get_data_driven_template_two_tasks(
        task1, task2, smoothing_fwhm=4,
        collection=1952, B=100, cap_subjects=False, n_jobs=1, seed=None):
    """
    Get (task1 - task2) data-driven template for two Neurovault contrasts

    Parameters
    ----------

    task1 : str
        Neurovault contrast
    task2 : str
        Neurovault contrast
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    collection : int
        Neurovault collection ID
    B : int
        number of permutations at training step
    cap_subjects : boolean
        If True, use only the first 15 subjects
    seed : int

    Returns
    -------

    pval0_quantiles : matrix of shape (B, p)
        Learned template (= sorted quantile curves)
    """
    fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm, collection=collection)
    if cap_subjects:
        # Let's compute the permuted p-values
        pval0 = sa.get_permuted_p_values_one_sample(fmri_input[:10, :],
                                                    B=B, seed=seed, n_jobs=n_jobs)
        # Sort to obtain valid template
        pval0_quantiles = np.sort(pval0, axis=0)
    else:
        # Let's compute the permuted p-values
        pval0 = sa.get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed, n_jobs=n_jobs)
        # Sort to obtain valid template
        pval0_quantiles = np.sort(pval0, axis=0)

    return pval0_quantiles


def get_processed_input(task1, task2, smoothing_fwhm=4, collection=1952):
    """
    Get (task1 - task2) processed input for a pair of Neurovault contrasts

    Parameters
    ----------

    task1 : str
        Neurovault contrast
    task2 : str
        Neurovault contrast
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    collection : int
        Neurovault collection ID

    Returns
    -------

    fmri_input : matrix of shape (n_subjects, p)
        Masked fMRI data
    nifti_masker :
        NiftiMasker object
    """
    # First, let's find the data and collect all the image paths
    data_path = get_data_dirs()[0]
    data_location_ = os.path.join(data_path, 'neurovault/collection_')
    data_location = data_location_ + str(collection)
    paths = [data_location + '/' + path for path in os.listdir(data_location)]

    files_id = []

    for path in paths:
        if path.endswith(".json") and 'collection_metadata' not in path:
            f = open(path)
            data = json.load(f)
            if 'relative_path' in data:
                files_id.append((data['relative_path'], data['file']))
            else:
                continue
    # Let's retain the images for the two tasks of interest
    # We also retain the subject name for each image file

    subjects1, subjects2 = [], []

    images_task1 = []
    for i in range(len(files_id)):
        if task1 in files_id[i][1]:
            img_path = files_id[i][0].split(sep=os.sep)[1]
            images_task1.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects1.append(filename.split(sep='base')[1])

    images_task1 = np.array(images_task1)

    images_task2 = []
    for i in range(len(files_id)):
        if task2 in files_id[i][1]:
            img_path = files_id[i][0].split(sep=os.sep)[1]
            images_task2.append(os.path.join(data_location, img_path))
            filename = files_id[i][1].split(sep='/')[6]
            subjects2.append(filename.split(sep='base')[1])

    images_task2 = np.array(images_task2)

    # Find subjects that appear in both tasks and retain corresponding indices

    common = sorted(list(set(subjects1) & set(subjects2)))
    indices1 = [subjects1.index(common[i]) for i in range(len(common))]
    indices2 = [subjects2.index(common[i]) for i in range(len(common))]

    # Mask and compute the difference between the two conditions

    nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
    all_imgs = np.concatenate([images_task1[indices1], images_task2[indices2]])
    nifti_masker.fit(all_imgs)
    fmri_input1 = nifti_masker.transform(images_task1[indices1])
    fmri_input2 = nifti_masker.transform(images_task2[indices2])

    fmri_input = fmri_input1 - fmri_input2

    return fmri_input, nifti_masker


def get_stat_img(task1, task2, smoothing_fwhm=4, collection=1952):
    """
    Get (task1 - task2) z-values map for two Neurovault contrasts

    Parameters
    ----------

    task1 : str
        Neurovault contrast
    task2 : str
        Neurovault contrast
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    collection : int
        Neurovault collection ID

    Returns
    -------

    z_vals_ :
        Unmasked z-values
    """
    fmri_input, nifti_masker = get_processed_input(
        task1, task2, smoothing_fwhm=smoothing_fwhm, collection=collection)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_vals_ = nifti_masker.inverse_transform(z_vals)

    return z_vals_


def calibrate_simes(fmri_input, alpha, k_max, B=100, n_jobs=1, seed=None):
    """
    Perform calibration using the Simes template

    Parameters
    ----------

    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    alpha : float
        Risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    n_jobs : int
        number of CPUs used for computation. Default = 1
    seed : int

    Returns
    -------

    pval0 : matrix of shape (B, p)
        Permuted p-values
    simes_thr : list of length k_max
        Calibrated Simes template
    """
    p = fmri_input.shape[1]  # number of voxels

    # Compute the permuted p-values
    pval0 = sa.get_permuted_p_values_one_sample(fmri_input,
                                                B=B,
                                                seed=seed,
                                                n_jobs=n_jobs)

    # Compute pivotal stats and alpha-level quantile
    piv_stat = sa.get_pivotal_stats(pval0, K=k_max)
    lambda_quant = np.quantile(piv_stat, alpha)

    # Compute chosen template
    simes_thr = sa.linear_template(lambda_quant, k_max, p)

    return pval0, simes_thr


def ari_inference(p_values, tdp, alpha, nifti_masker):
    """
    Find largest FDP controlling region using ARI.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    tdp : float
        True Discovery Proportion (= 1 - FDP)
    alpha : float
        Risk level
    nifti_masker: NiftiMasker
        masker used on current data

    Returns
    -------

    z_unmasked : nifti image of z_values of the FDP controlling region
    region_size_ARI : size of FDP controlling region

    """

    z_vals = norm.isf(p_values)
    hommel = _compute_hommel_value(z_vals, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    z_unmasked, region_size_ARI = sa.find_largest_region(p_values, ari_thr,
                                                         tdp,
                                                         nifti_masker)
    return z_unmasked, region_size_ARI


def bh_inference(p_values, fdr, masker=None):
    """
    Find largest FDR controlling region using BH.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    fdr : float
        False Discovery Rate
    masker: NiftiMasker
        masker used on current data

    Returns
    -------

    z_unmasked_cal : nifti image of z_values of the FDP controlling region
    region_size : size of FDR controlling region

    """
    z_map_ = norm.isf(p_values)

    z_cutoff = fdr_threshold(z_map_, fdr)

    region_size = len(z_map_[z_map_ > z_cutoff])

    if masker is not None:
        z_to_plot = np.copy(z_map_)
        z_to_plot[z_to_plot < z_cutoff] = 0
        z_unmasked_cal = masker.inverse_transform(z_to_plot)
        return z_unmasked_cal, region_size

    return region_size


def compute_bounds(task1s, task2s, learned_templates,
                   alpha, TDP, k_max, B,
                   smoothing_fwhm=4, n_jobs=1, seed=None):
    """
    Find largest FDP controlling regions on a list of contrast pairs
    using ARI, calibrated Simes and  learned templates.

    Parameters
    ----------

    task1s : list
        list of contrasts
    task2s : list
        list of contrasts
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    n_jobs : int
        number of CPUs used for computation. Default = 1

    Returns
    -------

    bounds_tot : matrix
        Size of largest FDP controlling regions for all three methods

    """

    simes_bounds = []
    learned_bounds = []
    ari_bounds = []

    for i in tqdm(range(len(task1s))):
        fmri_input, nifti_masker = get_processed_input(
                                                task1s[i], task2s[i],
                                                smoothing_fwhm=smoothing_fwhm)

        stats_, p_values = stats.ttest_1samp(fmri_input, 0)
        _, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)
        pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                           k_max=k_max, B=B,
                                           n_jobs=n_jobs, seed=seed)
        calibrated_tpl = sa.calibrate_jer(alpha, learned_templates,
                                          pval0, k_max)

        _, region_size_simes = sa.find_largest_region(p_values, simes_thr,
                                                      TDP,
                                                      nifti_masker)

        _, region_size_learned = sa.find_largest_region(p_values,
                                                        calibrated_tpl,
                                                        TDP,
                                                        nifti_masker)

        simes_bounds.append(region_size_simes)
        learned_bounds.append(region_size_learned)
        ari_bounds.append(region_size_ARI)

    bounds_tot = np.vstack([ari_bounds, simes_bounds, learned_bounds])
    return bounds_tot


def compute_bounds_single_task(task1s, task2s,
                               alpha, TDP, k_max, B,
                               smoothing_fwhm=4, n_jobs=1, seed=None):
    """
    Find largest FDP controlling regions for a single contrast pair
    using the Notip procedure on many different learned templates.

    Parameters
    ----------

    task1s : list
        list of contrasts
    task2s : list
        list of contrasts
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    n_jobs : int
        number of CPUs used for computation. Default = 1

    Returns
    -------

    bounds_tot : matrix
        Size of largest FDP controlling regions for all three methods

    """

    simes_bounds = []
    learned_bounds = []
    ari_bounds = []

    test_task1 = 'task001_look_negative_cue_vs_baseline'
    test_task2 = 'task001_look_negative_rating_vs_baseline'

    fmri_input, nifti_masker = get_processed_input(
                                        test_task1, test_task2,
                                        smoothing_fwhm=smoothing_fwhm)
    
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    _, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                k_max=k_max, B=B,
                                n_jobs=n_jobs, seed=seed)

    for i in tqdm(range(len(task1s))):
        fmri_input_train, nifti_masker_train = get_processed_input(task1s[i], task2s[i], smoothing_fwhm=smoothing_fwhm)
        _, p_values_train = stats.ttest_1samp(fmri_input_train, 0)
        _, region_size_ARI_train = ari_inference(p_values_train, TDP, alpha, nifti_masker_train)
        if region_size_ARI_train <= 25:
            continue
        learned_templates_ = sa.get_permuted_p_values_one_sample(fmri_input_train, B=B, seed=seed, n_jobs=n_jobs)
        # Sort to obtain valid template
        learned_templates = np.sort(learned_templates_, axis=0)
        calibrated_tpl = sa.calibrate_jer(alpha, learned_templates,
                                        pval0, k_max)

        _, region_size_simes = sa.find_largest_region(p_values, simes_thr,
                                                    TDP,
                                                    nifti_masker)

        _, region_size_learned = sa.find_largest_region(p_values,
                                                        calibrated_tpl,
                                                        TDP,
                                                        nifti_masker)
        
        ari_bounds.append(region_size_ARI)
        simes_bounds.append(region_size_simes)
        learned_bounds.append(region_size_learned)

    bounds_tot = np.vstack([ari_bounds, simes_bounds, learned_bounds])
    return bounds_tot


def sim_experiment_notip(dim, FWHM, pi0, sig_train, sig_test, fdr, alpha=0.05, n_train=5, n_test=5, train_on_same=False, repeats=10, B=10, n_jobs=1, seed=None):

    '''
    Check if the FDP is successfully controlled for a given number of experiments on simulated data
    '''
    np.random.seed(seed)

    fdp_ari = []
    fdp_simes = []
    fdp_learned = []
    #fdp_bh = []

    tdp_ari = []
    tdp_simes = []
    tdp_learned = []
    #tdp_bh = []

    k_max = int((dim**3)/50)
    #k_max = n_clusters
    if not train_on_same:
        X_train, _, _ = generate_data(dim, FWHM, pi0, scale=sig_train, nsubjects=2 * n_train)
        learned_template_ = sa.get_permuted_p_values_one_sample(X_train, B=B, n_jobs=n_jobs)
        learned_template = np.sort(learned_template_, axis=0)

    for trials in tqdm(range(repeats)):

        X_test, beta_true, nifti_masker = generate_data(dim, FWHM, pi0, scale=sig_test, nsubjects=2 * n_test)
        if len(beta_true) != dim**3:
            continue
        _, p_values = stats.ttest_1samp(X_test, 0)

        pval0, simes_thr = calibrate_simes(X_test, alpha,
                                           k_max=k_max, B=B,
                                           n_jobs=n_jobs, seed=seed)
        
        if train_on_same:
            learned_template_ = sa.get_permuted_p_values_one_sample(X_test, B=B, n_jobs=n_jobs)
            learned_template = np.sort(learned_template_, axis=0)

        
        calibrated_tpl = sa.calibrate_jer(alpha, learned_template,
                                          pval0, k_max)

        z_vals = norm.isf(p_values)
        hommel = _compute_hommel_value(z_vals, alpha)
        ari_thr = sa.linear_template(alpha, hommel, hommel)

        size_ari, cutoff_ari = sa.find_largest_region(p_values, ari_thr, 1 - fdr)
        fdp, tdp = report_fdp_tdp(p_values, cutoff_ari, beta_true, dim**3)
        fdp_ari.append(fdp)
        tdp_ari.append(tdp)

        size_simes, cutoff_simes = sa.find_largest_region(p_values, simes_thr, 1 - fdr)
        fdp, tdp = report_fdp_tdp(p_values, cutoff_simes, beta_true, dim**3)
        fdp_simes.append(fdp)
        tdp_simes.append(tdp)

        size_ko, cutoff_ko = sa.find_largest_region(p_values, calibrated_tpl, 1 - fdr)
        fdp, tdp = report_fdp_tdp(p_values, cutoff_ko, beta_true, dim**3)
        fdp_learned.append(fdp)
        tdp_learned.append(tdp)
    
    tdp_ari = np.array(tdp_ari)
    tdp_simes = np.array(tdp_simes)
    tdp_learned = np.array(tdp_learned)

    return fdp_ari, fdp_simes, fdp_learned, ((tdp_simes - tdp_ari)/tdp_ari) * 100, ((tdp_learned - tdp_ari)/tdp_ari) * 100, ((tdp_learned - tdp_simes)/tdp_simes) * 100
    # return fdp_ari, fdp_simes, fdp_learned, tdp_ari, tdp_simes, tdp_learned


def report_fdp_tdp(p_values, cutoff, beta_true, n_clusters):
        selected = np.where(p_values <= cutoff)[0]
        prediction = np.array([0] * n_clusters)
        prediction[selected] = 1
        conf = confusion_matrix(beta_true, prediction)
        tn, fp, fn, tp = conf.ravel()
        if fp + tp == 0:
            fdp = 0
            tdp = 0
        else:
            fdp = fp/(fp+tp)
            tdp = tp/np.sum(beta_true)

        return fdp, tdp


def get_clusters_table_TDP(stat_img, stat_threshold, fmri_input,
                           learned_templates, alpha=0.05,
                           k_max=1000, B=1000, cluster_threshold=None,
                           two_sided=False, min_distance=8., seed=None):
    """Creates pandas dataframe with img cluster statistics.
    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).
    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.
    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks, subpeaks and estimated TDP using three methods
        from thresholded `stat_img`. For binary clusters
        (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.
    """
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
            'TDP (ARI)', 'TDP (Calibrated Simes)', 'TDP (Learned)']
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold
    # print(cluster_threshold)
    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)

    stat_map_ = _safe_get_data(stat_img)
    # Perform calibration before thresholding
    stat_map_nonzero = stat_map_[stat_map_ != 0]
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=k_max, B=B, seed=seed)
    learned_thr = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
    )

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = _safe_get_data(stat_img, ensure_finite=True,
                              copy_data=(cluster_threshold is not None))
    # Define array for 6-connectivity, aka NN1 or "faces"
    conn_mat = np.zeros((3, 3, 3), int)
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1
    voxel_size = np.prod(stat_img.header.get_zooms())
    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using CDT
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                'Attention: No clusters with stat {0} than {1}'.format(
                    'higher' if sign == 1 else 'lower',
                    stat_threshold * sign,
                )
            )
            continue

        # Now re-label and create table
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask
            masked_data_ = masked_data[masked_data != 0]
            # Compute TDP bounds on cluster using our 3 methods
            cluster_p_values = norm.sf(masked_data_)
            ari_tdp = sa.min_tdp(cluster_p_values, ari_thr)
            simes_tdp = sa.min_tdp(cluster_p_values, simes_thr)
            learned_tdp = sa.min_tdp(cluster_p_values, learned_thr)
            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(
                masked_data,
                stat_img.affine,
                min_distance=min_distance,
            )
            subpeak_vals *= sign  # flip signs if necessary
            subpeak_xyz = np.asarray(
                coord_transform(
                    subpeak_ijk[:, 0],
                    subpeak_ijk[:, 1],
                    subpeak_ijk[:, 2],
                    stat_img.affine,
                )
            ).tolist()
            subpeak_xyz = np.array(subpeak_xyz).T

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            for subpeak in range(n_subpeaks):
                if subpeak == 0:
                    row = [
                        c_id + 1,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        "{0:.2f}".format(subpeak_vals[subpeak]),
                        cluster_size_mm,
                        "{0:.2f}".format(ari_tdp),
                        "{0:.2f}".format(simes_tdp),
                        "{0:.2f}".format(learned_tdp),
                    ]
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = '{0}{1}'.format(
                        c_id + 1,
                        ascii_lowercase[subpeak - 1],
                    )
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        "{0:.2f}".format(subpeak_vals[subpeak]),
                        '',
                        '',
                        '',
                        '',
                    ]
                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols, data=rows)

    return df


def get_clusters_table_with_TDP(stat_img, fmri_input, stat_threshold=3,
                                alpha=0.05,
                                k_max=1000, n_permutations=1000, cluster_threshold=None,
                                methods=['Notip'],
                                two_sided=False, min_distance=8., n_jobs=2, seed=None):
    """Creates pandas dataframe with img cluster statistics.
    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).
    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.
    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks, subpeaks and estimated TDP using three methods
        from thresholded `stat_img`. For binary clusters
        (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.
    """
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold
    # print(cluster_threshold)
    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)

    stat_map_ = _safe_get_data(stat_img)
    # Perform calibration before thresholding
    stat_map_nonzero = stat_map_[stat_map_ != 0]
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=k_max, B=n_permutations, seed=seed)
    learned_templates_ = sa.get_permuted_p_values_one_sample(fmri_input,
                                                             B=n_permutations,
                                                             n_jobs=n_jobs,
                                                             seed=None)
    learned_templates = np.sort(learned_templates_, axis=0)
    learned_thr = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
    )

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = _safe_get_data(stat_img, ensure_finite=True,
                              copy_data=(cluster_threshold is not None))
    # Define array for 6-connectivity, aka NN1 or "faces"
    conn_mat = np.zeros((3, 3, 3), int)
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1
    voxel_size = np.prod(stat_img.header.get_zooms())
    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using CDT
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                'Attention: No clusters with stat {0} than {1}'.format(
                    'higher' if sign == 1 else 'lower',
                    stat_threshold * sign,
                )
            )
            continue

        # Now re-label and create table
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask
            masked_data_ = masked_data[masked_data != 0]
            # Compute TDP bounds on cluster using our 3 methods
            cluster_p_values = norm.sf(masked_data_)
            ari_tdp = sa.min_tdp(cluster_p_values, ari_thr)
            simes_tdp = sa.min_tdp(cluster_p_values, simes_thr)
            learned_tdp = sa.min_tdp(cluster_p_values, learned_thr)
            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(
                masked_data,
                stat_img.affine,
                min_distance=min_distance,
            )
            subpeak_vals *= sign  # flip signs if necessary
            subpeak_xyz = np.asarray(
                coord_transform(
                    subpeak_ijk[:, 0],
                    subpeak_ijk[:, 1],
                    subpeak_ijk[:, 2],
                    stat_img.affine,
                )
            ).tolist()
            subpeak_xyz = np.array(subpeak_xyz).T

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            for subpeak in range(n_subpeaks):
                if subpeak == 0:
                        if methods == ['ARI', 'Notip']:
                            cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
                                    'TDP (ARI)', 'TDP (Notip)']
                            row = [
                                c_id + 1,
                                subpeak_xyz[subpeak, 0],
                                subpeak_xyz[subpeak, 1],
                                subpeak_xyz[subpeak, 2],
                                "{0:.2f}".format(subpeak_vals[subpeak]),
                                cluster_size_mm,
                                "{0:.2f}".format(ari_tdp),
                                "{0:.2f}".format(learned_tdp)]
                        else:
                            cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
                                    'TDP (Notip)']
                            row = [
                                c_id + 1,
                                subpeak_xyz[subpeak, 0],
                                subpeak_xyz[subpeak, 1],
                                subpeak_xyz[subpeak, 2],
                                "{0:.2f}".format(subpeak_vals[subpeak]),
                                cluster_size_mm,
                                "{0:.2f}".format(learned_tdp)]                           
                                    
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = '{0}{1}'.format(
                        c_id + 1,
                        ascii_lowercase[subpeak - 1],
                    )
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        "{0:.2f}".format(subpeak_vals[subpeak]),
                        '']
                    
                    row += [''] * len(methods)

                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols, data=rows)

    return df


def get_tdp_bound_notip(stat_img, fmri_input, cluster_mask,
                        alpha=0.05,
                        k_max=1000, n_permutations=1000, cluster_threshold=None,
                        two_sided=False, min_distance=8., n_jobs=2, seed=None):
    """Creates pandas dataframe with img cluster statistics.
    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).
    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.
    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks, subpeaks and estimated TDP using three methods
        from thresholded `stat_img`. For binary clusters
        (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.
    """
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
            'TDP (ARI)', 'TDP (Notip)']
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold
    # print(cluster_threshold)
    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)

    stat_map_ = _safe_get_data(stat_img)
    # Perform calibration before thresholding
    stat_map_nonzero = stat_map_[stat_map_ != 0]
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=k_max, B=n_permutations, seed=seed)
    learned_templates_ = sa.get_permuted_p_values_one_sample(fmri_input,
                                                             B=n_permutations,
                                                             n_jobs=n_jobs,
                                                             seed=None)
    learned_templates = np.sort(learned_templates_, axis=0)
    learned_thr = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = _safe_get_data(stat_img, ensure_finite=True)
                             
    voxel_size = np.prod(stat_img.header.get_zooms())
    no_clusters_found = True
    rows = []

    masked_data = stat_map * cluster_mask
    masked_data_ = masked_data[masked_data != 0]
    c_id = 0
    c_val = np.max(masked_data_)
            
    # Compute TDP bounds on cluster using our 3 methods
    cluster_p_values = norm.sf(masked_data_)
    ari_tdp = sa.min_tdp(cluster_p_values, ari_thr)
    simes_tdp = sa.min_tdp(cluster_p_values, simes_thr)
    learned_tdp = sa.min_tdp(cluster_p_values, learned_thr)

    stat_img_ = new_img_like(stat_img, masked_data)

    return learned_tdp, stat_img_


def _compute_hommel_value(z_vals, alpha, verbose=False):
    """Compute the All-Resolution Inference hommel-value"""
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_samples + (alpha - slope * n_samples) / slope)
    if verbose:
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            warnings.warn('"verbose" option requires the package Matplotlib.'
                          'Please install it using `pip install matplotlib`.')
        else:
            plt.figure()
            plt.plot(p_vals, 'o')
            plt.plot([n_samples - hommel_value, n_samples], [0, alpha])
            plt.plot([0, n_samples], [0, 0], 'k')
            plt.show(block=False)
    return np.minimum(hommel_value, n_samples)
