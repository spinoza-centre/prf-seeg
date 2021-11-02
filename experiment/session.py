#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import math
import h5py
import urllib
import time
import pylink
import numpy as np
from psychopy import logging
import scipy.stats as ss
from psychopy.visual import GratingStim
from psychopy.core import getTime
from psychopy import parallel

from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import FixationLines, FixationBullsEye
from trial import BarPassTrial, InstructionTrial, DummyWaiterTrial, EmptyBarPassTrial, OutroTrial

def _rotate_origin_only(x, y, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

class PRFBarPassSession(PylinkEyetrackerSession):
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on=True):
        """ Initializes StroopSession object. 

        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-stroop_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        """
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file,
                         eyetracker_on=eyetracker_on)  # initialize parent class!
        # stimulus materials
        stim_file_path = os.path.join(os.path.split(__file__)[0], 'stimuli', self.settings['stimuli'].get('bg_stim_h5file'))
        if not os.path.isfile(stim_file_path):
            logging.warn(f'Downloading stimulus file from figshare to {stim_file_path}')
            urllib.request.urlretrieve(self.settings['stimuli'].get('bg_stim_url'), stim_file_path)
        
        try:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)
            self.parallel_triggering = True
        except:
            logging.warn(f'Attempted import of Parallel Port failed')
            self.parallel_triggering = False
        
        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)
            
        self.create_stimuli()
        self.create_trials()
        self.create_fixation_mark_times()
    
    def create_fixation_mark_times(self):
        # twice too many events for safety
        nr_events = int(2 * self.total_time / self.settings['design'].get('minimal_ifi_duration'))
        exponentials = np.random.exponential(self.settings['design'].get('exponential_ifi_mean'),nr_events)
        gaussians = np.random.randn(nr_events) * self.settings['design'].get('gaussian_ifi_sd')
        offsets = np.ones(nr_events) * self.settings['design'].get('offset_ifi_duration')
        minimum = self.settings['design'].get('minimal_ifi_duration')

        self.fix_event_durations = exponentials + gaussians + offsets
        self.fix_event_durations = self.fix_event_durations[self.fix_event_durations>minimum] # shouldn't be too restrictive
        self.fix_event_durations[self.fix_event_durations < offsets] = self.settings['design'].get('minimal_ifi_duration')
        
        self.fix_event_times = np.cumsum(self.fix_event_durations) + self.settings['design'].get('start_duration')
        self.stimulus_changed = False
        self.last_fix_event = 0
        # np.savetxt(os.path.join(self.output_dir, self.output_str + '_fix_events.tsv'), self.fix_event_times, delimiter='\t')

        self.fix_event_responses = np.zeros((self.fix_event_times.shape[0], 3))
        self.fix_event_responses[:,1] = self.fix_event_times

    def create_stimuli(self):
        """create stimuli, both background bitmaps, and bar apertures
        """

        self.fixation = FixationBullsEye(win=self.win,
                                      circle_radius=self.settings['stimuli'].get(
                                          'stim_size_pixels'),
                                      color=(0.5, 0.5, 0.5, 1),
                                      **{'lineWidth':self.settings['stimuli'].get('outer_fix_linewidth')})

        self.report_fixation = FixationLines(win=self.win,
                                             circle_radius=self.settings['stimuli'].get(
                                                 'fix_radius')*2,
                                             color=self.settings['stimuli'].get('fix_color'),
                                             **{'lineWidth':self.settings['stimuli'].get('inner_fix_linewidth')})
                                             
        h5stimfile = h5py.File(os.path.join(os.path.split(__file__)[
                               0], 'stimuli', self.settings['stimuli'].get('bg_stim_h5file')), 'r')
        self.bg_images = -1 + np.array(h5stimfile.get(
            'stimuli')) / 128
        h5stimfile.close()

        self.image_bg_stims = [GratingStim(win=self.win,
                                           tex=bg_img,
                                           units='pix', 
                                           texRes=self.bg_images.shape[1],
                                           colorSpace='rgb',
                                           size=self.settings['stimuli'].get('stim_size_pixels'),
                                           interpolate=True)
                               for bg_img in self.bg_images]

        # draw all the bg stimuli once, before they are used in the trials
        for ibs in self.image_bg_stims:
            ibs.draw()
        self.win.flip()
        self.win.flip()
        
        # set up a bunch of stimulus aperture arrays, for the combinations of 
        # bar widths, refresh times, and directions
        nr_frames_bar_pass = int(self.settings['stimuli'].get(
            'refresh_rate') * self.settings['design'].get('bar_duration'))
        bar_directions = np.array(
            self.settings['stimuli'].get('bar_directions'))
        self.unique_bar_directions = np.unique(
            bar_directions[bar_directions >= 0])

        self.aperture_dict = {}
        with h5py.File(os.path.join(self.output_dir, self.output_str + '_apertures.h5'), 'w') as h5f:
            for bar_width in self.settings['stimuli'].get('bar_widths'):
                self.aperture_dict.update({bar_width: {}})
                for bar_refresh_time in self.settings['stimuli'].get('bar_refresh_times'):
                    self.aperture_dict[bar_width].update({bar_refresh_time: {}})
                    bar_refresh_frames = bar_refresh_time * self.settings['stimuli'].get('refresh_rate')
                    for bar_direction in self.unique_bar_directions:
                        these_apertures = self.create_apertures(n_mask_pixels=self.bg_images.shape[1],
                                                                bar_direction=bar_direction,
                                                                bar_width=bar_width,
                                                                nr_bar_steps=int(nr_frames_bar_pass/bar_refresh_frames))
                        self.aperture_dict[bar_width][bar_refresh_time].update({bar_direction: these_apertures})

                        # save to h5file
                        ds_name = f'{bar_width}_{bar_refresh_time}_{bar_direction}'
                        ds_name = ds_name.replace('.', 'x')
                        h5f.create_dataset(ds_name, data=these_apertures, compression=6)

    def create_apertures(self, n_mask_pixels, bar_direction, bar_width, nr_bar_steps):
        """[summary]

        Args:
            n_mask_pixels ([type]): [description]
            bar_direction ([type]): [description]
            bar_width ([type]): [description]
            nr_bar_steps ([type]): [description]

        Returns:
            [type]: [description]
        """
        # middle of bars
        bar_step_positions = np.linspace(-bar_width-1,
                                         1+bar_width, nr_bar_steps, endpoint=True)

        # circular aperture is there for everyone
        X, Y = np.meshgrid(np.linspace(-1, 1, n_mask_pixels, endpoint=True),
                           np.linspace(-1, 1, n_mask_pixels, endpoint=True))
        ecc = np.sqrt(X**2+Y**2)
        circular_aperture = ecc < self.settings['stimuli'].get('aperture_radius')

        # bar apertures
        X, Y = _rotate_origin_only(X, Y, np.deg2rad(bar_direction))

        op_apertures = np.zeros([nr_bar_steps] + list(X.shape), dtype=bool)
        for i, bsp in enumerate(bar_step_positions):
            op_apertures[i] = (X > (bsp-bar_width)) & (X < (bsp+bar_width))
            op_apertures[i] *= circular_aperture

        return op_apertures

    def bar_stimulus_lookup(self, bar_width, bar_direction, bar_refresh_time):
        """bar_stimulus_lookup creates a lookup-table for 
        which apertures and which bg images will be shown when during this bar pass. 
        They index different number of items, since the refresh of the bar and the bg
        are divorced in this experiment. 

        Args:
            bar_width (float): width of bar in fraction relative to stim size
            bar_direction (float): orientation of bar in degrees
            bar_refresh_time (int): nr of display frames for which a single bar is shown
        """
        bar_display_frames = np.arange(
            self.aperture_dict[bar_width][bar_refresh_time][bar_direction].shape[0])
        nr_bg_frames_par_pass = int(self.settings['design'].get('bar_duration') / self.settings['stimuli'].get('bg_stim_refresh_time'))
            
        # the following ensures that subsequent frames of the bg do not accidentally repeat
        bg_stim_frames = np.mod(
            np.cumsum(
                np.random.randint(1, len(self.image_bg_stims)-2, size=nr_bg_frames_par_pass+2)),
            len(self.image_bg_stims))

        return {'bar_display_frames': bar_display_frames, 'bg_stim_frames': bg_stim_frames}

    def create_trials(self):
        """ Creates trials (ideally before running your session!) """

        instruction_trial = InstructionTrial(session=self,
                                             trial_nr=0,
                                             phase_durations=[np.inf],
                                             txt=self.settings['stimuli'].get('instruction_text'),
                                             keys=['space'], 
                                             draw_each_frame=False)

        dummy_trial = DummyWaiterTrial(session=self,
                                       trial_nr=1,
                                       phase_durations=[
                                       np.inf, self.settings['design'].get('start_duration')],
                                       txt=self.settings['stimuli'].get('pretrigger_text'), 
                                       draw_each_frame=False)

        bar_directions = np.array(
            self.settings['stimuli'].get('bar_directions'))
        bar_widths = np.array(
            self.settings['stimuli'].get('bar_widths'))
        bar_refresh_times = np.array(
            self.settings['stimuli'].get('bar_refresh_times'))

        # random ordering for bar parameters
        bws, brts = np.meshgrid(bar_widths, bar_refresh_times)
        bws, brts = bws.ravel(), brts.ravel()
        bar_par_order = np.arange(bws.shape[0])
        np.random.shuffle(bar_par_order)
        bar_par_counter = 0

        self.trials = [instruction_trial, dummy_trial]
        trial_counter = 2
        start_time = self.settings['design'].get('start_duration')
        for i in range(len(bar_widths)):
            for j in range(len(bar_refresh_times)):
                for k, bd in enumerate(bar_directions):
                    if bar_directions[k] < 0:  # no bar during these, they are blanks
                        bw = -1
                        brt = 1    
                        phase_durations = [self.settings['design'].get('blank_duration')]
                    else:
                        bw = bws[bar_par_order[bar_par_counter]]
                        brt = brts[bar_par_order[bar_par_counter]]
                        phase_durations = [self.settings['design'].get('bar_duration')]
                    parameters = {'bar_width': bw,
                                  'bar_refresh_time': brt,
                                  'bar_direction': bd,
                                  'start_time': start_time,
                                  'bg_stim_refresh_time': self.settings['stimuli'].get('bg_stim_refresh_time')}
                    if bd == -1:                        
                        self.trials.append(EmptyBarPassTrial(
                        session=self,
                        trial_nr=trial_counter,
                        phase_durations=phase_durations,
                        phase_names=['stim'],
                        parameters=parameters,
                        timing='seconds',
                        verbose=True, 
                        draw_each_frame=False))
                    else:

                        blt = self.bar_stimulus_lookup(bar_width=bw,
                                            bar_direction=bd,
                                            bar_refresh_time=brt)
                        self.trials.append(BarPassTrial(
                            session=self,
                            trial_nr=trial_counter,
                            phase_durations=phase_durations,
                            phase_names=['stim'],
                            parameters=parameters,
                            timing='seconds',
                            aperture_sequence=blt['bar_display_frames'],
                            bg_img_sequence=blt['bg_stim_frames'],
                            verbose=True, 
                            draw_each_frame=False))

                    trial_counter = trial_counter + 1
                    start_time = start_time + phase_durations[0]
                bar_par_counter = bar_par_counter + 1

        outro_trial = OutroTrial(session=self,
                                 trial_nr=trial_counter,
                                 phase_durations=[
                                     self.settings['design'].get('end_duration')],
                                 txt='', 
                                 draw_each_frame=False)
        
        self.trials.append(outro_trial)
        self.total_time = start_time + self.settings['design'].get('end_duration')

    def create_trial(self):
        pass

    def run(self):
        """ Runs experiment. """
        # self.create_trials()  # create them *before* running!

        if self.eyetracker_on:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()

    def close(self):
        h5_seq_file = os.path.join(self.output_dir, self.output_str + '_seq_timing.h5')
        for trial in self.trials:
            if type(trial) == BarPassTrial:
                trial.bg_img_sequence_df.to_hdf(h5_seq_file, key=f'trial_{str(trial.trial_nr).zfill(3)}/bg_imgs', mode='a')
                trial.aperture_sequence_df.to_hdf(h5_seq_file, key=f'trial_{str(trial.trial_nr).zfill(3)}/apertures', mode='a')                
        
        # save out and calculate first-pass behavioral results
        t = getTime() - self.experiment_start_time
        true_fix_events = self.fix_event_responses[self.fix_event_responses[:,1] < t]
        np.savetxt(os.path.join(self.output_dir, self.output_str + '_fix_responses.tsv'), 
                    true_fix_events, delimiter='\t')
        responded_events = true_fix_events[:,0] != 0
        perc_caught = responded_events.sum() / true_fix_events.shape[0]
        mean_rt = true_fix_events[responded_events,2].mean()
        print(f'Percentage caught: {perc_caught}, Mean Reaction Time: {mean_rt}')

        super().close()  # close parent class!

    def parallel_trigger(self, trigger):
        if self.parallel_triggering:
            self.port.setData(trigger)
            time.sleep(self.settings['design'].get('ttl_trigger_delay'))
            self.port.setData(0)
            time.sleep(self.settings['design'].get('ttl_trigger_delay'))
            # P = windll.inpoutx64
            # P.Out32(0x0378, self.settings['design'].get('ttl_trigger_blank')) # send the event code (could be 1-20)
            # time.sleep(self.settings['design'].get('ttl_trigger_delay')) # wait for 1 ms for receiving the code
            # P.Out32(0x0378, 0) # send a code to clear the register
            # time.sleep(self.settings['design'].get('ttl_trigger_delay')) # wait for 1 ms"""
        else:
            logging.warn(f'Would have sent trigger {trigger}')

