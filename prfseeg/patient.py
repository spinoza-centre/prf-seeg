import os
import mne
import yaml
import json
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt

from inspect import getsourcefile

from .acquisition import Acquisition2kHz, Acquisition10kHz

# quick function for coreg checking


def plot_overlay(image, compare, title, thresh=None):
    """Define a helper function for comparing plots."""
    image = nb.orientations.apply_orientation(
        np.asarray(image.dataobj), nb.orientations.axcodes2ornt(
            nb.orientations.aff2axcodes(image.affine))).astype(np.float32)
    compare = nb.orientations.apply_orientation(
        np.asarray(compare.dataobj), nb.orientations.axcodes2ornt(
            nb.orientations.aff2axcodes(compare.affine))).astype(np.float32)
    if thresh is not None:
        compare[compare < np.quantile(compare, thresh)] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(np.take(image, [image.shape[i] // 2], axis=i).squeeze().T,
                  cmap='gray')
        ax.imshow(np.take(compare, [compare.shape[i] // 2],
                          axis=i).squeeze().T, cmap='gist_heat', alpha=0.5)
        ax.invert_yaxis()
        ax.axis('off')
    fig.tight_layout()


class Patient:
    """Patient is a single patient, with electrodes in fixed positions, 
    containing multiple runs of sEEG data as well as pre-op T1w and post-op CT anatomical images
    """
    # instance attributes

    def __init__(self, subject, raw_dir, derivatives_dir):
        """[summary]

        Args:
            subject ([type]): [description]
            raw_dir ([type]): [description]
            derivatives_dir ([type]): [description]
        """
        self.subject = subject
        self.raw_dir = raw_dir
        self.derivatives_dir = derivatives_dir

        self.raw_func_dir = os.path.join(raw_dir, self.subject, 'func')
        self.raw_anat_dir = os.path.join(raw_dir, self.subject, 'anat')

        self.preprocessing_dir = os.path.join(
            derivatives_dir, 'prep', self.subject, 'func')
        self.localization_dir = os.path.join(
            derivatives_dir, 'prep', self.subject, 'loc')
        self.tfr_dir = os.path.join(
            derivatives_dir, 'tfr', self.subject, 'func')
        self.prf_dir = os.path.join(
            derivatives_dir, 'pRF', self.subject, 'func')
        self.subjects_dir = os.path.join(derivatives_dir, 'freesurfer')

        for d in (self.preprocessing_dir, self.localization_dir, self.tfr_dir, self.prf_dir):
            os.makedirs(d, exist_ok=True)
        self.filepath = os.path.abspath(getsourcefile(lambda: 0))
        with open(os.path.join(os.path.split(os.path.split(self.filepath)[0])[0], 'analysis', 'config.yml'), 'r') as yf:
            self.analysis_settings = yaml.safe_load(yf)

    def __repr__(self):
        return f'Patient "{self.subject}" at "{self.raw_dir}", derivatives at {self.derivatives_dir}'

    # instance method
    def gather_acquisitions(self):
        self.acquisitions = []
        for run, acq in zip(range(1, self.analysis_settings['nr_runs']+1), self.analysis_settings['acquisition_types']):
            if acq == '2kHz':
                this_run = Acquisition2kHz(raw_dir=self.raw_func_dir,
                                           run_nr=run,
                                           patient=self,
                                           task=self.analysis_settings['task'])
            elif acq == '10kHz':
                this_run = Acquisition10kHz(raw_dir=self.raw_func_dir,
                                            run_nr=run,
                                            patient=self,
                                            task=self.analysis_settings['task'])
            self.acquisitions.append(this_run)

    def preprocess(self):
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        for acq in self.acquisitions:
            preprocessed_fn = acq.raw_filename.replace(
                'bids', 'derivatives/prep').replace('.edf', '_ieeg.fif.gz')
            acq.notch_resample_cut(
                resample_frequency=self.analysis_settings['preprocessing']['downsample_frequency'],
                notch_filter_frequencies=self.analysis_settings['preprocessing']['notch_frequencies'],
                raw_file_name=None,
                output_file_name=preprocessed_fn)
            acq.preprocessed_fn = preprocessed_fn
            tfr_fn = preprocessed_fn.replace(
                'prep', 'tfr').replace('.fif.gz', '.h5')
            acq.tfr(raw_file_name=acq.preprocessed_fn,
                    tfr_logspace_low=self.analysis_settings['preprocessing']['tfr_logspace_low'],
                    tfr_logspace_high=self.analysis_settings['preprocessing']['tfr_logspace_high'],
                    tfr_logspace_nr=self.analysis_settings['preprocessing']['tfr_logspace_nr'],
                    tfr_subsampling_factor=self.analysis_settings[
                        'preprocessing']['tfr_subsampling_factor'],
                    output_filename=tfr_fn)
            acq.tfr_fn = tfr_fn

    def find_electrode_positions(self, which_run=0):
        # 1. check if freesurfer has run
        # 2. run MNE coregistration
        # 3. save electrode positions in stereotypical format
        # follow: https://mne.tools/stable/auto_tutorials/clinical/10_ieeg_localize.html#sphx-glr-auto-tutorials-clinical-10-ieeg-localize-py
        # and: https://mne.tools/stable/auto_tutorials/clinical/20_seeg.html
        """
        # on the server:
        mkdir /tank/shared/tmp/prf-seeg &&
        singularity run --cleanenv -B /tank -B /scratch /tank/shared/software/bids_apps/fmriprep-20.2.6.simg             --skip_bids_validation --participant-label 002 --anat-only --nthreads 30 --omp-nthreads 30 --fs-license-file /tank/shared/software/freesurfer_dev/license.txt --notrack -w /tank/shared/tmp/prf-seeg /scratch/2021/prf-seeg/data/bids/ /scratch/2021/prf-seeg/data/derivatives participant 
        """
        CT_orig = nb.load(os.path.join(
            self.raw_anat_dir, f'{self.subject}_CT.nii.gz'))
        T1w_orig = nb.load(os.path.join(
            self.raw_anat_dir, f'{self.subject}_T1w.nii.gz'))
        T1w_FS = nb.load(os.path.join(self.subjects_dir,
                         self.subject, 'mri', 'T1.mgz'))

        if not os.path.isfile(T1w_FS):
            print(
                'No FreeSurfer T1w file for this subject found. Please run FreeSurfer first. ')
            return

        reg_affine, _ = mne.transforms.compute_volume_registration(
            CT_orig, T1w_FS, pipeline='rigids')
        CT_aligned = mne.transforms.apply_volume_registration(
            CT_orig, T1w_FS, reg_affine)

        self.gather_acquisitions()
        self.acquisitions[which_run]._read_raw()
        # use estimated `trans` which was used when the locations were found previously
        subj_trans = mne.coreg.estimate_head_mri_t(subject=self.subject,
                                                   subjects_dir=self.subjects_dir)
        subj_mni_fiducials = mne.coreg.get_mni_fiducials(subject=self.subject,
                                                         subjects_dir=self.subjects_dir)
        gui = mne.gui.locate_ieeg(self.acquisitions[which_run].raw.info,
                                  subj_trans,
                                  CT_aligned,
                                  subject=self.subject,
                                  subjects_dir=self.subjects_dir)
