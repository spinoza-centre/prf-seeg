# prf-seeg
sEEG experiment and analysis


# To Do

- set the channel types using MNE's raw object syntax, i.e. `self.raw.set_channel_types()`
- the following sensor types are accepted: 
    ```ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, dbs, stim, syst, ecog, hbo, hbr, fnirs_cw_amplitude, fnirs_fd_ac_amplitude, fnirs_fd_phase, fnirs_od```
- then re-reference using those