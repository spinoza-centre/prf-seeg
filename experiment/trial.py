#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math, time
import numpy as np
import pandas as pd

from exptools2.core import Trial
from psychopy.core import getTime
from psychopy.visual import TextStim
from psychopy import logging

from stimuli import FixationLines

# #### Windows triggering
# try:
#     from ctypes import windll
#     win_triggering = True
# except ImportError as error:
#     logging.warn(f'Attempted import of windll failed, {error.__class__.__name__}: {error}')
#     win_triggering = False

class BarPassTrial(Trial):
    
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, aperture_sequence, bg_img_sequence,
                 verbose=True, draw_each_frame=False):
        """ Initializes a BarPassTrial object. 
        
        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        trial_nr: int
            Trial nr of trial
        phase_durations : array-like
            List/tuple/array with phase durations
        phase_names : array-like
            List/tuple/array with names for phases (only for logging),
            optional (if None, all are named 'stim')
        parameters : dict
            Dict of parameters that needs to be added to the log of this trial
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        verbose : bool
            Whether to print extra output (mostly timing info)
        """
        super().__init__(session, trial_nr, phase_durations, phase_names,
                         parameters, timing, load_next_during_phase=None, verbose=verbose, draw_each_frame=draw_each_frame)
        # print(self.parameters)
        # internalize these sequences and their expected times in the trials

        expected_aperture_times = self.parameters['start_time'] + np.arange(len(aperture_sequence)+1) * self.parameters['bar_refresh_time']
        expected_bg_img_times = self.parameters['start_time'] + np.arange(len(bg_img_sequence)+1) * self.parameters['bg_stim_refresh_time']

        self.aperture_sequence_df = pd.DataFrame(np.array([np.r_[aperture_sequence, 0], 
                                            expected_aperture_times, 
                                            np.zeros_like(expected_aperture_times)]).T, 
                                                columns=['seq_index', 'expected_time', 'empirical_time'])
        self.bg_img_sequence_df = pd.DataFrame(np.array([np.r_[bg_img_sequence, 0], 
                                            expected_bg_img_times, 
                                            np.zeros_like(expected_bg_img_times)]).T, 
                                                columns=['seq_index', 'expected_time', 'empirical_time'])

        self.aperture_masks = self.session.aperture_dict[self.parameters['bar_width']][self.parameters['bar_refresh_time']][self.parameters['bar_direction']]

        self.bg_display_frame = -1
        self.bar_display_frame = -1
    
    def run(self):

        #####################################################
        ## TRIGGER HERE
        #####################################################
        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_bar'))

        super().run()  # run parent class!

    def draw(self):

        draw = False

        total_display_time = (getTime() - self.session.experiment_start_time)
        trial_display_time = total_display_time - self.parameters['start_time']
        bg_display_frame = math.floor(trial_display_time / self.session.settings['stimuli'].get('bg_stim_refresh_time'))
        if bg_display_frame != self.bg_display_frame:
            self.bg_display_frame = bg_display_frame
            self.bg_img_sequence_df['empirical_time'].loc[bg_display_frame] = total_display_time
            draw = True

        # find and fill in the binary mask
        bar_display_frame = np.min([int(trial_display_time / self.parameters['bar_refresh_time']), self.aperture_masks.shape[0]])
        if bar_display_frame != self.bar_display_frame:
            self.bar_display_frame = bar_display_frame
            self.aperture_sequence_df['empirical_time'].loc[bar_display_frame] = total_display_time
            draw = True

        if draw:
            if total_display_time > self.session.fix_event_times[self.session.last_fix_event]:
                self.session.last_fix_event = self.session.last_fix_event + 1
                self.session.report_fixation.setColor(-1 * self.session.report_fixation.color)

            # identify stimulus object, and decide whether to draw
            if math.fmod(trial_display_time, self.parameters['bar_blank_interval']) > self.parameters['bar_blank_duration']:
                which_bg_stim = self.session.image_bg_stims[int(self.bg_img_sequence_df['seq_index'].loc[bg_display_frame])]

                which_mask = np.min([self.aperture_sequence_df['seq_index'].loc[bar_display_frame], 
                                    self.aperture_masks.shape[0]])

                mask = self.aperture_masks[int(which_mask)]
                which_bg_stim.mask = (mask * 2) - 1
                which_bg_stim.draw()
            
            self.session.fixation.draw()
            self.session.report_fixation.draw()
            self.session.win.flip()


    def get_events(self):
        events = super().get_events()
        if len(events) > 0:
            t = getTime() - self.session.experiment_start_time
            # discard early events
            if self.session.fix_event_times[0] > t:
                pass
            else:
                which_last_fix_event = np.arange(self.session.fix_event_times.shape[0])[self.session.fix_event_times < t][-1]
                self.session.fix_event_responses[which_last_fix_event][0] = t
                self.session.fix_event_responses[which_last_fix_event][2] = t - self.session.fix_event_times[which_last_fix_event]

class EmptyBarPassTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None, draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)
    
    def draw(self):
        total_display_time = (getTime() - self.session.experiment_start_time)
        trial_display_time = total_display_time - self.parameters['start_time']

        if total_display_time > self.session.fix_event_times[self.session.last_fix_event]:
            self.session.last_fix_event = self.session.last_fix_event + 1
            self.session.report_fixation.setColor(-1 * self.session.report_fixation.color)

        self.session.fixation.draw()
        self.session.report_fixation.draw()
        self.session.win.flip()

    def run(self):

        #####################################################
        ## TRIGGER HERE
        #####################################################
        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_blank'))

        super().run()  # run parent class!

    def get_events(self):
        events = super().get_events()
        if len(events) > 0:
            t = getTime() - self.session.experiment_start_time
            # discard early events
            if self.session.fix_event_times[0] > t:
                pass
            else:
                which_last_fix_event = np.arange(self.session.fix_event_times.shape[0])[self.session.fix_event_times < t][-1]
                self.session.fix_event_responses[which_last_fix_event][0] = t
                self.session.fix_event_responses[which_last_fix_event][2] = t - self.session.fix_event_times[which_last_fix_event]

class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        text_position_x = self.session.settings['various'].get('text_position_x')
        text_position_y = self.session.settings['various'].get('text_position_y')

        if txt is None:
            txt = '''Press any button to continue.'''

        self.text = TextStim(self.session.win, txt,
                             height=txt_height, 
                             wrapWidth=txt_width, 
                             pos=[text_position_x, text_position_y],
                             font='Songti SC',
                             alignText = 'center',
                             anchorHoriz = 'center',
                             anchorVert = 'center')
        self.text.setSize(txt_height)

        self.keys = keys

    def draw(self):
        self.session.fixation.draw()
        self.session.report_fixation.draw()

        self.text.draw()
        self.session.win.flip()

    def get_events(self):
        events = super().get_events()

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()


class DummyWaiterTrial(InstructionTrial):
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None,
                 txt="Waiting for scanner triggers.", draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, txt, draw_each_frame=draw_each_frame, **kwargs)
    
    def draw(self):
        self.session.fixation.draw()
        if self.phase == 0:
            self.text.draw()
        else:
            self.session.report_fixation.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    if self.phase == 0:
                        self.stop_phase()
                        self.session.win.flip()
                        #####################################################
                        ## TRIGGER HERE
                        #####################################################
                        self.session.experiment_start_time = getTime()
                        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_start'))


class OutroTrial(InstructionTrial):
    """ Simple trial with only fixation cross.  """

    def __init__(self, session, trial_nr, phase_durations, txt='', draw_each_frame=False, **kwargs):

        txt = ''''''
        super().__init__(session, trial_nr, phase_durations, txt=txt, draw_each_frame=draw_each_frame, **kwargs)

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == 'space':
                    self.stop_phase()        