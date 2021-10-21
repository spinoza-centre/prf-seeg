import os
import math
import h5py
import numpy as np
import scipy.stats as ss
from psychopy.visual import GratingStim

from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import FixationLines
from trial import BarPassTrial, InstructionTrial, DummyWaiterTrial, OutroTrial


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
        self.n_trials = self.settings['design'].get('n_trials')

        self.fixation = FixationLines(win=self.win,
                                      circle_radius=self.settings['stimuli'].get(
                                          'aperture_radius')*2,
                                      color=(1, -1, -1))

        self.report_fixation = FixationLines(win=self.win,
                                             circle_radius=self.settings['stimuli'].get(
                                                 'fix_radius')*2,
                                             color=self.settings['stimuli'].get('fix_color'))
        
        self.create_stimuli()

    def create_stimuli(self):
        """create stimuli, both background bitmaps, and bar apertures
        """

        h5stimfile = h5py.File(os.path.join(os.path.split(__file__)[
                               0], self.settings['stimuli'].get('bg_stim_h5file')), 'r')
        self.bg_images = -1 + np.array(h5stimfile.get(
            'stimuli')) / 128
        h5stimfile.close()
        print(self.bg_images.shape)
        self.image_bg_stims = [GratingStim(win=self.win,
                                           tex=bg_img,
                                           units='pix',
                                           colorSpace='rgb',
                                           size=self.settings['stimuli'].get('stim_size_pixels'))
                               for bg_img in self.bg_images]
        # self.image_bg_stims.extend([GratingStim(win=self.win,
        #                                     tex=bg_img[:,::-1],
        #                                     units='pix',
        #                                     size=self.settings['stimuli'].get('stim_size_pixels')) for bg_img in self.bg_images])

        # set up a bunch of stimulus aperture arrays
        nr_frames_bar_pass = self.settings['stimuli'].get(
            'refresh_rate') * self.settings['design'].get('bar_duration')
        bar_directions = np.array(
            self.settings['stimuli'].get('bar_directions'))
        self.unique_bar_directions = np.unique(
            bar_directions[bar_directions >= 0])
        # self.unique_bar_directions = self.unique_bar_directions[
        #     self.unique_bar_directions < 180]
        self.aperture_dict = {}
        for bar_width in self.settings['stimuli'].get('bar_widths'):
            self.aperture_dict.update({bar_width: {}})
            for bar_refresh_frames in self.settings['stimuli'].get('bar_refresh_frames'):
                self.aperture_dict[bar_width].update({bar_refresh_frames: {}})
                for bar_direction in self.unique_bar_directions:
                    self.aperture_dict[bar_width][bar_refresh_frames].update({bar_direction:
                                                                              self.create_apertures(
                                                                                  n_mask_pixels=self.settings['stimuli'].get(
                                                                                      'stim_size_pixels'),
                                                                                  bar_direction=bar_direction,
                                                                                  bar_width=bar_width,
                                                                                  nr_bar_steps=int(nr_frames_bar_pass/bar_refresh_frames)
                                                                              )
                                                                              })

    def create_apertures(self, n_mask_pixels, bar_direction, bar_width, nr_bar_steps):
        """create apertures from settings.
        """
        # middle of bars
        bar_step_positions = np.linspace(-bar_width/2,
                                         1+bar_width/2, nr_bar_steps, endpoint=True)

        # circular aperture is there for everyone
        X, Y = np.meshgrid(np.linspace(-1, 1, n_mask_pixels, endpoint=True),
                           np.linspace(-1, 1, n_mask_pixels, endpoint=True))
        ecc = np.sqrt(X**2+Y**2)
        circular_aperture = ecc < (2*self.settings['stimuli'].get(
            'aperture_radius') / n_mask_pixels)

        # bar apertures
        X, Y = _rotate_origin_only(X, Y, np.deg2rad(bar_direction))

        op_apertures = np.zeros([nr_bar_steps] + list(X.shape), dtype=bool)
        for i, bsp in enumerate(bar_step_positions):
            op_apertures[i] = (X > (bsp-bar_width/2)) & (X < (bsp+bar_width/2))
            op_apertures[i] *= circular_aperture

        return op_apertures

    def bar_stimulus_lookup(self, bar_width, bar_direction, bar_refresh_frames):
        """bar_stimulus_lookup creates a lookup-table for 
        which apertures and which bg images will be shown when during this bar pass. 
        They index different number of items, since the refresh of the bar and the bg
        are divorced in this experiment. 

        Args:
            bar_width (float): width of bar in fraction relative to stim size
            bar_direction (float): orientation of bar in degrees
            bar_refresh_frames (int): nr of display frames for which a single bar is shown
        """
        bar_frames = np.arange(
            self.aperture_dict[bar_width][bar_refresh_frames][bar_direction].shape[0])
        nr_display_frames_bar = self.settings['stimuli'].get(
            'refresh_rate') * self.settings['design'].get('bar_duration')
        nr_frames_bar_pass = nr_display_frames_bar / \
            self.settings['stimuli'].get('bg_stim_refresh_frames')
        # the following ensures that subsequent frames of the bg do not accidentally repeat
        bg_stim_frames = np.mod(
            np.cumsum(
                np.random.randint(1, len(self.image_bg_stims)-2, size=nr_frames_bar_pass)),
            len(self.image_bg_stims)-1) + 1

        return {'bar_frames': bar_frames, 'bg_stim_frames': bg_stim_frames}

    def create_trials(self):
        """ Creates trials (ideally before running your session!) """

        instruction_trial = InstructionTrial(session=self,
                                             trial_nr=0,
                                             phase_durations=[np.inf],
                                             txt='Please keep fixating at the center.',
                                             keys=['space'])

        dummy_trial = DummyWaiterTrial(session=self,
                                       trial_nr=1,
                                       phase_durations=[
                                           np.inf, self.settings['design'].get('start_duration')],
                                       txt='Waiting for experiment to start')

        bar_directions = np.array(
            self.settings['stimuli'].get('bar_directions'))
        bar_widths = np.array(
            self.settings['stimuli'].get('bar_widths'))
        bar_refresh_frames = np.array(
            self.settings['stimuli'].get('bar_refresh_frames'))

        # random ordering for bar parameters
        bwi = np.array([np.random.choice(np.arange(len(bar_widths)))
                        for x in range(len(bar_refresh_frames))])
        brfi = np.array([np.random.choice(np.arange(len(bar_refresh_frames)))
                         for x in range(len(bar_widths))])

        self.trials = [instruction_trial, dummy_trial]
        trial_counter = 2
        start_time = 0
        for i in range(len(bar_widths)):
            for j in range(len(bar_refresh_frames)):
                for k, bd in enumerate(bar_directions):
                    if bar_directions[k] < 0:  # no bar during these, they are blanks
                        bw = -1
                        brf = 1
                        phase_durations = [
                            self.settings['design'].get('blank_duration')]
                    else:
                        bw = bar_widths[bwi[j, i]],
                        brf = bar_refresh_frames[brfi[i, j]]
                        phase_durations = [
                            self.settings['design'].get('bar_duration')]
                    parameters = {'bar_width': bw,
                                  'bar_refresh_frames': brf,
                                  'bar_direction': bd,
                                  'start_time': start_time}
                    parameters.update(self.bar_stimulus_lookup(bar_width=bw,
                                                               bar_direction=bd,
                                                               bar_refresh_frames=brf))

                    self.trials.append(BarPassTrial(
                        session=self,
                        trial_nr=trial_counter,
                        phase_durations=phase_durations,
                        phase_names=['stim'],
                        parameters=parameters,
                        timing='seconds',
                        verbose=True))

                    trial_counter = trial_counter + 1
                    start_time = start_time + phase_durations[0]

        outro_trial = OutroTrial(session=self,
                                 trial_nr=trial_counter,
                                 phase_durations=[
                                     self.settings['design'].get('end_duration')],
                                 txt='')

        self.trials.append(outro_trial)

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
