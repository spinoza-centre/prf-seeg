import math
import numpy as np
from exptools2.core import Trial
from psychopy.core import getTime
from psychopy.visual import TextStim
from stimuli import FixationLines

class BarPassTrial(Trial):
    
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, 
                 verbose=True):
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
                         parameters, timing, load_next_during_phase=None, verbose=verbose)
        # print(self.parameters)
    
    def run(self):

        #####################################################
        ## TRIGGER HERE
        #####################################################

        super().run()  # run parent class!


    def draw(self):

        total_display_time = (getTime() - self.session.experiment_start_time)
        trial_display_time = total_display_time - self.parameters['start_time']
        bg_display_frame = math.floor(trial_display_time / self.session.settings['stimuli'].get('bg_stim_refresh_time'))
        # stimulus object
        bg_stim = self.session.image_bg_stims[self.parameters['bg_stim_frames'][bg_display_frame]]
        # fill in the binary mask
        bar_display_frame = int(trial_display_time / self.parameters['bar_refresh_time'])

        which_mask = np.min([self.parameters['bar_display_frames'][bar_display_frame], self.session.aperture_dict[self.parameters['bar_width']][self.parameters['bar_refresh_time']][self.parameters['bar_direction']].shape[0]])

        bg_stim.mask = self.session.aperture_dict[self.parameters['bar_width']][self.parameters['bar_refresh_time']][self.parameters['bar_direction']][which_mask]
        
        bg_stim.draw()
        
        if total_display_time > self.session.fix_event_times[self.session.last_fix_event]:
            self.session.last_fix_event = self.session.last_fix_event + 1
            self.session.report_fixation.setColor(-1 * self.session.report_fixation.color)

        self.session.fixation.draw()
        self.session.report_fixation.draw()

    # def get_events(self):
    #     events = super().get_events()
    #     pass

class EmptyBarPassTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None, **kwargs):

        super().__init__(session, trial_nr, phase_durations, **kwargs)
    
    def draw(self):
        total_display_time = (getTime() - self.session.experiment_start_time)
        trial_display_time = total_display_time - self.parameters['start_time']

        if total_display_time > self.session.fix_event_times[self.session.last_fix_event]:
            self.session.last_fix_event = self.session.last_fix_event + 1
            self.session.report_fixation.setColor(-1 * self.session.report_fixation.color)

        self.session.fixation.draw()
        self.session.report_fixation.draw()

    # def get_events(self):
    #     events = super().get_events()
    #     pass

class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, **kwargs):

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        if txt is None:
            txt = '''Press any button to continue.'''

        self.text = TextStim(self.session.win, txt,
                             height=txt_height, wrapWidth=txt_width)

        self.keys = keys

    def draw(self):
        self.session.fixation.draw()
        self.session.report_fixation.draw()

        self.text.draw()

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
                 txt="Waiting for scanner triggers.", **kwargs):

        super().__init__(session, trial_nr, phase_durations, txt, **kwargs)
    
    def draw(self):
        self.session.fixation.draw()
        if self.phase == 0:
            self.text.draw()
        else:
            self.session.report_fixation.draw()

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

class OutroTrial(InstructionTrial):
    """ Simple trial with only fixation cross.  """

    def __init__(self, session, trial_nr, phase_durations, txt='', **kwargs):

        txt = ''''''
        super().__init__(session, trial_nr, phase_durations, txt=txt, **kwargs)

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == 'space':
                    self.stop_phase()        