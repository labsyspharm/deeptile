# Major conclusion from this experiment
1. scikit-image.external.tifffile is roughly as fast as tifffile
2. Loading 3 channels took ~80 sec; yet loading 1 channel only took ~20 sec.
Load each channel one by one is likely faster.
