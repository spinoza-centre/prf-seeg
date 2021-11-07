import os
import mne
import yaml
import json
import numpy as np
import scipy as sp
import pandas as pd


class Patient:
    """Patient is a single patient, with electrodes in fixed positions, containing multiple runs of sEEG data as well as pre-op T1w and post-op CT anatomical images
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
        self.raw_func_dir = os.path.join(raw_dir, self.subject, 'func')
        self.raw_anat_dir = os.path.join(raw_dir, self.subject, 'anat')

        with os.path.join(os.path.split(os.path.split(__file__)[0])[0], 'analysis', 'config.yml') as file:
            self.analysis_settings = yaml.safe_load(file)

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
    """Acquisition is a single sEEG run, with associated metadata. 
    It is the vehicle for preprocessing.
    """    
    def __init__(self, raw_dir, run_nr, acq, patient, task='pRF'):
        """[summary]

        Args:
            raw_dir ([type]): [description]
            run_nr ([type]): [description]
            acq ([type]): [description]
            patient ([type]): [description]
            task (str, optional): [description]. Defaults to 'pRF'.
        """        
        self.raw_dir = raw_dir
        self.run_nr = run_nr
        self.acq = acq
        self.patient = patient
        self.task = task

        self.raw_filename = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_acq-{self.acq}.edf')

        self.evt_tsv_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_events.tsv')
        self.fix_tsv_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_fix_responses.tsv')
        self.exp_yml_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_expsettings.yml')
        self.seq_timing_h5_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_seq_timing.h5')
        self.aperture_h5_file = os.path.join(self.raw_dir, self.patient.subject, 'func', f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_apertures.h5')

        self.raw = None

    def _read_raw(self, raw_file_name=None):
        """_read_raw internalizes a specific file as self.raw
        This makes self.raw a flexible reference, because it can be used before or after preprocessing.

        Args:
            raw_file_name (string, filepath, optional): path to raw .edf or .fif filename to be used. 
                            Defaults to None, which takes the original raw_filename from the construction.
        """        
        if raw_file_name == None:
            self.raw = mne.io.read_raw_edf(self.raw_filename)
        else:
            self.raw = mne.io.read_raw_edf(raw_file_name)
    
    def internalize_metadata(self):
        """internalize_metadata reads in the metadata from various file formats associated with this run.
        """        
        # expt yaml file
        with open(self.exp_yml_file, 'r') as file:
            self.experiment_settings = yaml.safe_load(file)
        self.raw_fixation_events = pd.DataFrame(np.loadtxt(self.fix_tsv_file, sep='\t'), columns=['time_since_t', 'fix_event_times', 'initial_reaction_time'])
        # read seq timings and aperture/bg stim data from h5 files.
        self.bg_img_timing = self._read_expt_h5(self.seq_timing_h5_file, 'bg_imgs')
        self.aperture_timing = self._read_expt_h5(self.aperture_h5_file, 'apertures')

        # events from the experimental .tsv file.
        evt_data = pd.read_csv(self.evt_tsv_file, sep='\t')
        start_time = evt_data[evt_data.event_type == 'pulse'].onset_abs.values[0]
        self.tsv_trial_data = evt_data[evt_data.event_type == 'stim'][['bar_direction', 'bar_width', 'bar_refresh_time', 'onset_abs']]
        self.tsv_trial_data = self.tsv_trial_data[~np.isnan(self.tsv_trial_data['bar_width'].values)]
        self.tsv_trial_data['onset_time'] = self.tsv_trial_data.onset_abs - start_time
        self.tsv_trial_data.reindex(index=np.arange(self.tsv_trial_data.shape[0]))
        self.tsv_trial_data.reset_index(inplace=True)
        
    def _read_expt_h5(self, h5file, folder):
        """_read_expt_h5 reads the hdf5 files which are in standardized format.

        Args:
            h5file (str, path): the source hdf5 file.
            folder (str): name of the folder/array in which the data are stored.

        Returns:
            dictionary: dictionary, keys: trials, values: pandas DataFrames
        """        
        # get items from h5 file
        with h5py.File(h5file, 'r') as f:
            ks = list(f.keys())

        return {k: pd.read_hdf(h5file, key=f'{k}/{folder}', mode='r') for k in ks}

    def identify_triggers(self, raw_file_name=None):
        if self.raw == None:
            self._read_raw(raw_file_name)

        dc_channels_indx = np.array([True if 'DC' in ch else False for ch in self.raw.ch_names])
        dc_channels = np.array(self.raw.ch_names)[np.array(dc_channels_indx)]

        dc_data = self.raw.get_data(picks=dc_channels)

        self.bar_trigger_idx = np.argsort(dc_data.var(1))[-1]
        self.bar_trigger_ch = dc_channels[self.bar_trigger_idx]
        self.blank_trigger_idx = np.argsort(dc_data.var(1))[-2]
        self.blank_trigger_ch = dc_channels[self.blank_trigger_idx]

        self.blank_onsets, self.blank_onset_indx = self._timings_from_trig_channel(dc_data[self.blank_trigger_idx])
        self.bar_onsets, self.bar_onset_indx = self._timings_from_trig_channel(dc_data[self.bar_trigger_idx])

        self.bar_onset_indx = np.setdiff1d(self.bar_onset_indx, self.blank_onset_indx)

    def _timings_from_trig_channel(trig_channel_data, ll=4.0):
        """_timings_from_trig_channel takes a trigger channel's recording and finds the onset times and indices

        Args:
            trig_channel_data (numpy.ndarray): samples of the trigger channel
            ll (float, optional): Threshold for designating event occurrence in multiples of sds from the channel's mean, 
                                i.e. the value determines sensitivity. Defaults to 4.

        Returns:
            (onsets, onset_indx) tuple: onsets in the timeframe of the channel's samples, and their integer indices. 
        """        
        baseline, spread = np.median(trig_channel_data), np.std(trig_channel_data)
        onsets = np.r_[np.diff(trig_channel_data > baseline+ll*spread)>0, False]
        onset_indx = np.arange(trig_channel_data.shape[0])[onsets]
        return onsets, onset_indx

    def sync_eeg_behavior(self):
        """find eeg trigger samples with each of the 'trials' in the trial data from the experiment
        """        
        # create events from EEG traces
        tdf = pd.DataFrame(np.r_[self.bar_onset_indx.astype(int), self.blank_onset_indx.astype(int)], columns=['eeg_tt'])
        tdf['category'] = ['bar' for bi in self.bar_onset_indx]+['blank' for bi in self.blank_onset_indx]
        # all of the following to delete the first empty non-trial, and
        # connect to the .tsv derived data.
        tdf = tdf.sort_values(by='eeg_tt')
        tdf.reset_index(inplace=True)
        tdf.drop(0, inplace=True)
        tdf.reset_index(inplace=True)
        tdf.reindex(index=np.arange(trial_data.shape[0]))

        # Now we can use the info in self.trial_data
        self.trial_data = pd.concat((self.tsv_trial_data, tdf[['eeg_tt', 'category']]), axis=1)

    def notch_resample_cut(self, 
                           resample_frequency=1000, 
                           notch_filter_frequencies=[50,100,150,200,250], 
                           raw_file_name=None,
                           output_file_name=None):
        if self.raw == None:
            self._read_raw(raw_file_name)
        # 1. notch filter
        # 2. resample
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass
        self.raw.load_data()
        self.raw.notch_filter(notch_filter_frequencies, picks=self.non_signal_channels)
        self.raw.resample(resample_frequency, stim_picks=self.trigger_channels)

        self.identify_triggers()
        self.sync_eeg_behavior()

        first_sample = self.trial_data.eeg_tt.iloc[0]
        last_sample = self.trial_data.eeg_tt.iloc[-1] \
            + self.experiment_settings['design']['blank_duration'] * resample_frequency
        if output_file_name != None:
            self.raw.save(fname=output_file_name,
                          tmin=first_sample,
                          tmax=last_sample)
        

    def tfr(self, 
            sample_frequency=1000, 
            tfr_frequencies=[]):
        """[summary]

        Args:
            resample_frequency (int, optional): [description]. Defaults to 1000.
            notch_filter_frequencies (list, optional): [description]. Defaults to [50,100,150,200,250].
        """            
        # 1. resample
        # 2. notch filter
        # 3. t0 at 't' press
        # 4. tfr from t0 to end of last bar pass

        tfr_array_multitaper(raw_data_np[np.newaxis, ...], sfreq=sample_frequency, freqs=freqs, n_jobs=4, decim=5, output='power')


        return    

    def split_to_pRF_runs(self, stage='tfr'):
        """[summary]

        Args:
            stage (str, optional): [description]. Defaults to 'tfr'.
        """        
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
