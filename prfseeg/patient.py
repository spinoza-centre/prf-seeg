import os
import mne
import yaml
import json
import numpy as np
import scipy as sp
import pandas as pd

from acquisition import Acquisition2kHz, Acquisition10kHz

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
        self.derivatives_dir = self.derivatives_dir

        self.raw_func_dir = os.path.join(raw_dir, self.subject, 'func')
        self.raw_anat_dir = os.path.join(raw_dir, self.subject, 'anat')

        self.preprocessing_dir = os.path.join(derivatives_dir, 'prep', self.subject, 'func')
        self.localization_dir = os.path.join(derivatives_dir, 'prep', self.subject, 'loc')
        self.tfr_dir = os.path.join(derivatives_dir, 'tfr', self.subject)
        self.prf_dir = os.path.join(derivatives_dir, 'pRF', self.subject)

        for d in (self.preprocessing_dir, self.localization_dir, self.tfr_dir, self.prf_dir):
            os.makedirs(d, exist_ok=True)
        with os.path.join(os.path.split(os.path.split(__file__)[0])[0], 'analysis', 'config.yml') as file:
            self.analysis_settings = yaml.safe_load(file)

    # instance method
    def gather_acquisitions(self):
        self.acquisitions = []
        for run, acq in zip(range(1, self.analysis_settings['nr_runs']+1), self.analysis_settings['acquisition_types']):
            if acq == '2kHz:
                this_run = Acquisition2kHz(raw_dir=self.raw_func_dir, 
                                            run_nr=run, 
                                            acq=acq, 
                                            patient=self, 
                                            task=self.analysis_settings['task'])
            elif acq == '10kHz':
                this_run = Acquisition10kHz(raw_dir=self.raw_func_dir, 
                                            run_nr=run, 
                                            acq=acq, 
                                            patient=self, 
                                            task=self.analysis_settings['task'])
            self.acquisitions.append(this_run)
            

    def preprocess(self):
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        pass

    def find_electrode_positions(args):
        # 1. check if freesurfer has run
        # 2. run MNE coregistration 
        # 3. save electrode positions in stereotypical format
        # follow: https://mne.tools/stable/auto_tutorials/clinical/10_ieeg_localize.html#sphx-glr-auto-tutorials-clinical-10-ieeg-localize-py
        # and: https://mne.tools/stable/auto_tutorials/clinical/20_seeg.html
        pass
