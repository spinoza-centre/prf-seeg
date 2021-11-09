import os
import mne
import yaml, json, h5py
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_multitaper

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

        self.raw_filename = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_acq-{self.acq}.edf')

        self.evt_tsv_file = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_events.tsv')
        self.fix_tsv_file = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_fix_responses.tsv')
        self.exp_yml_file = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_expsettings.yml')
        self.seq_timing_h5_file = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_seq_timing.h5')
        self.aperture_h5_file = os.path.join(self.raw_dir, f'{self.patient.subject}_run-{str(self.run_nr).zfill(2)}_task-{self.task}_apertures.h5')

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
        self.raw_fixation_events = pd.DataFrame(np.loadtxt(self.fix_tsv_file, delimiter='\t'), columns=['time_since_t', 'fix_event_times', 'initial_reaction_time'])
        # read seq timings and aperture/bg stim data from h5 files.
        self.bg_img_timing = self._read_bg_imgs_h5(self.seq_timing_h5_file, 'bg_imgs')
        self.aperture_timing = self._read_apertures_h5(self.aperture_h5_file, 'apertures')

        # events from the experimental .tsv file.
        evt_data = pd.read_csv(self.evt_tsv_file, sep='\t')
        start_time = evt_data[evt_data.event_type == 'pulse'].onset_abs.values[0]
        self.tsv_trial_data = evt_data[evt_data.event_type == 'stim'][['bar_direction', 'bar_width', 'bar_refresh_time', 'onset_abs']]
        self.tsv_trial_data = self.tsv_trial_data[~np.isnan(self.tsv_trial_data['bar_width'].values)]
        self.tsv_trial_data['onset_time'] = self.tsv_trial_data.onset_abs - start_time
        self.tsv_trial_data.reindex(index=np.arange(self.tsv_trial_data.shape[0]))
        self.tsv_trial_data.reset_index(inplace=True)
        
    def _read_bg_imgs_h5(self, h5file, folder):
        """_read_bg_imgs_h5 reads the hdf5 files which are in standardized format.

        Args:
            h5file (str, path): the source hdf5 file.
            folder (str): name of the folder/array in which the data are stored.

        Returns:
            dictionary: dictionary, keys: trials, values: pandas DataFrames
        """        
        # get items from h5 file
        with h5py.File(h5file, 'r') as f:
            ks = list(f.keys())

        ops = {}
        for k in ks:
            try:
                ops.update({k: pd.read_hdf(h5file, key=f"{k.replace('.','x')}/{folder}", mode='r') })
            except KeyError as err:
                print(err)
        return ops

    def _read_apertures_h5(self, h5file, folder):
        """_read_apertures_h5 reads the hdf5 files which are in standardized format.

        Args:
            h5file (str, path): the source hdf5 file.
            folder (str): name of the folder/array in which the data are stored.

        Returns:
            dictionary: dictionary, keys: trials, values: pandas DataFrames
        """        
        # get items from h5 file
        with h5py.File(h5file, 'r') as f:
            ks = list(f.keys())
            ops = {k: np.array(f.get(k.replace('.','x')), dtype=bool) for k in ks}
        return ops

    def identify_triggers(self, raw_file_name=None):
        if self.raw == None:
            self._read_raw(raw_file_name)

        dc_channels_indx = np.array([True if self.patient.analysis_settings['trigger_channels'] in ch else False for ch in self.raw.ch_names])
        dc_channels = np.array(self.raw.ch_names)[np.array(dc_channels_indx)]

        dc_data = self.raw.get_data(picks=dc_channels)

        self.bar_trigger_idx = np.argsort(dc_data.var(1))[-1]
        self.bar_trigger_ch = dc_channels[self.bar_trigger_idx]
        self.blank_trigger_idx = np.argsort(dc_data.var(1))[-2]
        self.blank_trigger_ch = dc_channels[self.blank_trigger_idx]

        self.blank_onsets, self.blank_onset_indx = self._timings_from_trig_channel(dc_data[self.blank_trigger_idx])
        self.bar_onsets, self.bar_onset_indx = self._timings_from_trig_channel(dc_data[self.bar_trigger_idx])

        self.bar_onset_indx = np.setdiff1d(self.bar_onset_indx, self.blank_onset_indx)

    def _timings_from_trig_channel(self, trig_channel_data, ll=4.0):
        """_timings_from_trig_channel takes a trigger channel's recording and finds the onset times and indices

        Args:
            trig_channel_data (numpy.ndarray): samples of the trigger channel
            ll (float, optional): Threshold for designating event occurrence in multiples of sds from the channel's mean, 
                                i.e. the value determines sensitivity. Defaults to 4.

        Returns:
            (onsets, onset_indx) tuple: onsets in the timeframe of the channel's samples, and their integer indices. 
        """        
        spread = np.std(trig_channel_data)
        spread = np.std(trig_channel_data)
        baseline = np.median(trig_channel_data)
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
        tdf.reindex(index=np.arange(self.tsv_trial_data.shape[0]))

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
        self._get_data_channels()
        self.raw.load_data()
        mne.set_eeg_reference(self.raw, verbose=True)

        self.raw.notch_filter(notch_filter_frequencies, picks=self.signal_channels)
        self.raw.resample(resample_frequency, stim_picks=self.trigger_channels)

        if output_file_name != None:
            # now, we need to cut the data up for saving:
            self.internalize_metadata()
            self.identify_triggers()
            self.sync_eeg_behavior()

            first_sample = self.trial_data.eeg_tt.iloc[0] 
            last_sample = self.trial_data.eeg_tt.iloc[-1] \
                + (self.experiment_settings['design']['blank_duration'] * resample_frequency) / self.raw.info['sfreq']
            
            # convert from index to time
            first_time = first_sample / self.raw.info['sfreq']
            last_time = last_sample / self.raw.info['sfreq']

            self.raw.save(fname=output_file_name,
                          tmin=first_time,
                          tmax=last_time,
                          overwrite=True)
        
    def _get_data_channels(self, 
            raw_file_name=None):
        if self.raw == None:
            self._read_raw(raw_file_name)

        dc_channels_indx = np.array([True if 'DC' in ch else False for ch in self.raw.ch_names])
        self.dc_channel_names = np.array(self.raw.ch_names)[np.array(dc_channels_indx)]

        self.data_channel_bools = np.array([np.array([False if ndc in rcn else True 
                                                for ndc in self.patient.analysis_settings['non_data_channels']]).prod(dtype=bool) 
                                                    for rcn in self.raw.ch_names])
        self.data_channel_names = np.array(self.raw.ch_names)[self.data_channel_bools]
        self.signal_channels = np.arange(len(self.data_channel_bools))[self.data_channel_bools]
        self.trigger_channels = np.arange(len(self.raw.ch_names))[dc_channels_indx]

    def tfr(self, 
            raw_file_name=None, 
            tfr_logspace_low=0.1,
            tfr_logspace_high=2.5,
            tfr_logspace_nr=200,
            tfr_subsampling_factor=5,
            output_filename=None):
        if self.raw == None:
            self._read_raw(raw_file_name)

        self._get_data_channels()
        # assumes that input data are t0 to end of last bar pass blank
        raw_data_np = self.raw.get_data(picks=self.data_channel_names, 
                                        start=0)

        freqs = np.logspace(tfr_logspace_low, tfr_logspace_high, tfr_logspace_nr)
        tfr_data = tfr_array_multitaper(raw_data_np[np.newaxis, ...], 
                            sfreq=self.raw.info['sfreq'], 
                            freqs=freqs, 
                            n_jobs=self.patient.analysis_settings['nr_jobs'], 
                            decim=tfr_subsampling_factor, 
                            output='power')

        if output_filename != None:
            with h5py.File(output_filename, 'a') as h5f:
                h5f.create_dataset('freqs', data=freqs, compression=6)
                h5f.create_dataset('np_data', data=raw_data_np, compression=6)
                h5f.create_dataset('tfr_data', data=tfr_data, compression=6)
            ch_names_df = pd.DataFrame(np.arange(len(self.data_channel_names)), index=self.data_channel_names)
            ch_names_df.to_hdf(output_filename, key='ch_names', mode='a')

            

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
       super().__init__(raw_dir, run_nr, acq='2kHz', patient=patient, task='pRF')
       print(f'Acquisition2kHz {run} for {patient} on task {task} created.')

class Acquisition10kHz(Acquisition):
    def __init__(self, raw_dir, run_nr, patient, task='pRF'):
       # call super() function
       super().__init__(raw_dir, run_nr, acq='10kHz', patient=patient, task='pRF')
       print(f'Acquisition10kHz {run} for {patient} on task {task} created.')

class PRF_run:
    def __init__(self, bar_width, bar_refresh_time, bar_directions, bar_duration, blank_duration, bg_stim_array, aperture_array):
        print('initing')
