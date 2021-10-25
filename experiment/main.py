#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import os.path as op
from psychopy import logging
from session import PRFBarPassSession

# from psychopy import prefs
# #change the pref libraty to PTB and set the latency mode to high precision
# prefs.hardware['audioLib'] = 'PTB'
# prefs.hardware['audioLatencyMode'] = 3

parser = argparse.ArgumentParser(description='A Population Receptive Field experiment')
parser.add_argument('subject', default=None, nargs='?', 
                    help='the subject of the experiment, as a zero-filled integer, such as 001, or 04.')
parser.add_argument('run', default=0, type=int, nargs='?', 
                    help='the run nr of the experimental run, an integer, such as 1, or 99.')
parser.add_argument('eyelink', default=0, type=int, nargs='?')

cmd_args = parser.parse_args()
subject, run, eyelink = cmd_args.subject, cmd_args.run, cmd_args.eyelink

if subject is None:
    subject = 999

if eyelink == 1:
    eyetracker_on = True
    logging.warn("Using eyetracker")
else:
    eyetracker_on = False
    logging.warn("Using NO eyetracker")

output_str = f'sub-{subject}_run-{str(run).zfill(2)}_task-pRF'
settings_fn = op.join(op.dirname(__file__), 'settings.yml')

session_object = PRFBarPassSession(output_str=output_str,
                        output_dir=None,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
session_object.run()
session_object.close()