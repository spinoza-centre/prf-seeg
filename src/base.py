import os, mne, yaml, json
import numpy as np
import scipy as sp
import pandas as pd


class Patient:

    # instance attributes
    def __init__(self, subject, raw_dir, derivatives_dir):
       self.subject = subject
       self.raw_func_dir = os.path.join(raw_dir, self.subject, 'func')
       self.raw_anat_dir = os.path.join(raw_dir, self.subject, 'anat')

    # instance method
    def gather_raw_files(self, raw_dir):
       return

    def preprocess(self, raw_dir):
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        return

    def find_electrode_positions(args):
        # 1. check if freesurfer has run
        # 2. run MNE coregistration 
        # 3. save electrode positions in stereotypical format
        # follow: https://mne.tools/stable/auto_tutorials/clinical/10_ieeg_localize.html#sphx-glr-auto-tutorials-clinical-10-ieeg-localize-py
        # and: https://mne.tools/stable/auto_tutorials/clinical/20_seeg.html
        pass

class Acquisition:
    def __init__(self, raw_dir, run_nr, acq, patient, task='pRF'):
        self.raw_dir = raw_dir
        self.run_nr = run_nr
        self.acq = acq
        self.patient = patient
        self.task = task

        self.raw_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_acq-{self.acq}.edf')

        self.evt_tsv_file = f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_events.tsv'
        self.fix_tsv_file = f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_fix_responses.tsv'
        self.exp_yml_file = f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_expsettings.yml'
        self.seq_timing_h5_file = f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_seq_timing.h5'
        self.aperture_h5_file = f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_apertures.h5'

        self.raw = mne.io.read_raw_edf(self.raw_file)
    
    def internalize_metadata(self):
        # expt yaml file
        with open(self.exp_yml_file, 'r') as file:
            self.experiment_settings = yaml.safe_load(file)
        self.raw_fixation_events = pd.DataFrame(np.loadtxt(self.fix_tsv_file, sep='\t'), columns=['time_since_t', 'fix_event_times', 'initial_reaction_time'])
        self.events_from_tsv = pd.read_csv(self.evt_tsv_file, sep='\t')
        # read seq timings and aperture/bg stim data from h5 files.
        self.bg_img_timing = self._read_expt_h5(self.seq_timing_h5_file, 'bg_imgs')
        self.aperture_timing = self._read_expt_h5(self.aperture_h5_file, 'apertures')
        
    def _read_expt_h5(self, h5file, folder):
        # get items from h5 file
        with h5py.File(h5file, 'r') as f:
            ks = list(f.keys())

        return {k: pd.read_hdf(h5file, key=f'{k}/{folder}', mode='r') for k in ks}


    def resample_and_notch(self, 
                           resample_frequency=1000, 
                           notch_filter_frequencies=[50,100,150,200,250]):
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        return

    def tfr(self, 
            resample_frequency=1000, 
            notch_filter_frequencies=[50,100,150,200,250]):
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        return    

    def split_to_pRF_runs(self, stage='tfr'):
        # 1. find triggers
        # 2. find parameters for each pRF run
        # 3. 

class Acquisition2kHz(Acquisition):
    def __init__(self, raw_dir, run_nr, patient, task='pRF'):
       # call super() function
       super().__init__(raw_dir, run_nr, acq='2kHz', patient, task='pRF')
       print('Acquisition2kHz is ready')

    def preprocess(self):
        pass

class Acquisition10kHz(Acquisition):
    def __init__(self, raw_dir, run_nr, patient, task='pRF'):
       # call super() function
       super().__init__(raw_dir, run_nr, acq='10kHz', patient, task='pRF')
       print('Acquisition10kHz is ready')

class PRF_run:
    def __init__(self, bar_width, bar_refresh_time, bar_directions, bar_duration, blank_duration, bg_stim_array, aperture_array):
        print('initing')