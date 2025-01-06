# radar-signal-processing
Files Overview
## 1. statistical analysis and visualization
This script performs statistical analysis and visualization of radar data.

Functionalities:

Imports radar signal data from .mat files.

Calculates:

1. Mean signal power for each range gate.
   
2. Median signal power for each range gate.
   
3. Standard Deviation (STD) of signal power for each range gate.
   
4. Visualizes these metrics for two receivers.
   
5. Key Visualizations:

Line plots showing the mean, median, and STD values of signal power as a function of range.

## 2. Coherent integration

This script focuses on coherent integration and spectral analysis of radar data.

Functionalities:

1. Implements Coherent Integration to enhance signal clarity and reduce noise.
2. Performs Fast Fourier Transform (FFT) on Raw radar data, Coherently integrated radar data.
3. Compares time-domain and frequency-domain data before and after coherent integration.

## 3. auto-correlation, signal smoothing

This script handles auto-correlation, signal smoothing, and FFT analysis.

Functionalities:

1. Computes auto-correlation of radar data to estimate signal strength.
2. Smoothens data using correlation lengths for range gates.
3. Performs FFT on raw and smoothed radar data to analyze spectral characteristics.
Key Visualizations:

## 4. Doppler velocities

This script analyzes Doppler velocities and converts radar data from spherical to Cartesian coordinates.

Functionalities:

1. Converts spherical coordinates to Cartesian coordinates for velocity analysis.
2. Calculates: Horizontal velocities (North-South, East-West), Vertical velocities (upward), Smoothens velocity data to remove noise ,Removes outliers using a standard deviation threshold.
Key Visualizations:
