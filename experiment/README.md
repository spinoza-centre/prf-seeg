# Population Receptive Field Experiment

This code implements a pRF experiment in python, built on top of the exptools2 package for experimentation. It is modeled after the HCP retinotopy experiment, and uses its stimulus materials for the texture in the pRF bars.The code is tested to have reasonable non-drift timing accuracy. It saves out the exact sequence of textures and bar apetures used during the experiment -- including their timing during the experiment, as well as BIDS-compliant `.tsv` event files and a timing diagnostics figure. 

Here's a sample, example youtube movie of a single run of the experiment is [here](https://youtu.be/FmrIJlZ7o6c):

https://user-images.githubusercontent.com/436593/138601665-8af3687b-334e-4240-b772-a5c9bf4e3fc9.mp4

## Running the experiment

After installation (see below), running the experiment is just running a few lines in the command line:

```
conda activate exptools
python main.py 1 2 0
```

This will run the second run of the experiment for the first subject, without eyetracking, whereas:

```
conda activate exptools
python main.py 2 1 1
```

will run the first run of the experiment for the second subject, with eyetracking on an SR Research eyelink. There's more information in the `argparse` output, and it's adaptable in `main.py`

## Customizing the experiment

At present (and this is subject to change) the experiment runs multiple 'standard' pRF bar pass design experiments in a single run, and expects a Full HD (1920x1080) screen. Timing is defined in seconds, so it can be run equally well on a 60Hz or 120 Hz screen. Spatial parameters are defined in pixels, so calculating subtense of stimulation in the visual field field is left as an exercise to the user :)

The settings that determine the specific parameters of the experiment are in `settings.yml`, and can be adapted to change things like timing and spatial parameters of the experiment. 

## Installation

This repository contains only python code and settings. It downloads the large `.hdf5` file of stimulus materials from figshare when run the first time (so be patient dependent your connection's speed). 

Here’s a short description of how it should be set up. 

1. Install Anaconda/Miniconda on the relevant computer
2. Follow the instructions to install our package exptools2 from the README here: [exptools2](https://github.com/VU-Cog-Sci/exptools2#installation-using-conda), installing a virtual conda environment containing all the required code.
3. You need to do an additional `conda install h5py`
4. Then download the relevant SR Research pylink library, instructions are here: [https://www.psychopy.org/api/hardware/pylink.html](https://www.psychopy.org/api/hardware/pylink.html)
5. Then, clone the experiment repo: `git clone https://github.com/spinoza-centre/prf-seeg.git`
6. Navigate to its folder and run the experiment as above. 
