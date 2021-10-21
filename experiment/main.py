import os.path as op
import argparse
from psychopy import logging

from session import PRFBarPassSession

parser = argparse.ArgumentParser()
parser.add_argument('subject', default=None, nargs='?')
parser.add_argument('run', default=None, nargs='?')
parser.add_argument('eyelink', default=False, nargs='?')

cmd_args = parser.parse_args()
subject, run, eyelink = cmd_args.subject, cmd_args.run, cmd_args.eyelink

if subject is None:
    subject = input('Subject? (999): ')
    subject = 999 if subject == '' else subject

if run is None:
    run = input('Run? (None): ')
    run = 0 if run == '' else run
elif run == '0':
    run = 0

if eyelink:
    eyetracker_on = True
    logging.warn("Using eyetracker")
else:
    eyetracker_on = False
    logging.warn("Using NO eyetracker")


output_str = f'sub-{subject}_run-{run}_task-BP'
settings_fn = op.join(op.dirname(__file__), 'settings.yml')

session_object = PRFBarPassSession(output_str=output_str,
                        output_dir=None,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
session_object.create_trials()
logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
session_object.run()
session_object.close()