# Phase Noise Estimation from Ngspice Transient Output

This repository contains a Python script to estimate **single-sideband (SSB) phase noise** from **transient simulation data**, such as that generated by **Ngspice**. The method applies FFT and Power Spectral Density (PSD) analysis to voltage vs. time data, enabling post-processing of oscillator phase noise at arbitrary frequency offsets.

## 📌 Features

- Reads `.txt` files with alternating time and voltage values
- Applies a Hanning window with proper RMS normalization
- Computes FFT and PSD (in V²/Hz and dBV²/Hz)
- Locates the carrier peak and evaluates phase noise at **±1 MHz offset**
- Selects the **worst-case (higher PSD)** sideband
- Plots both **FFT magnitude** and **PSD** with annotations
- Reports phase noise in **dBc/Hz**

## 🧠 Algorithm Overview

### 1. Preprocessing
- Remove DC offset
- Apply a Hanning window
- Compute the RMS correction gain for amplitude normalization

### 2. FFT and PSD Calculation
- Normalize FFT magnitude:  
  `|V(f)| = (2 / (N * G_rms)) * |FFT(v_windowed)|`

- Compute one-sided Power Spectral Density:  
  `PSD(f) = |V(f)|² / Δf`

Where:
- `|V(f)|` is the normalized FFT magnitude  
- `Δf` is the FFT frequency resolution  
- `G_rms` is the RMS gain of the Hanning window

### 3. Carrier and Offset Analysis
- Find the carrier peak (maximum FFT magnitude)
- Compute total carrier power from 3-bin PSD sum:  
  `Carrier Power = (PSD[peak-1] + PSD[peak] + PSD[peak+1]) * Δf`
- Measure PSD at ±1 MHz and compute SSB phase noise:  
  `L(f_offset) = 10 * log10(PSD_offset / Carrier Power)`  
  (in dBc/Hz)

### 4. Plotting
- FFT Magnitude Spectrum (in Volts)
- PSD (in dBV²/Hz) with peak and offset annotations

**A data file named "example_1G.ngspice" has been uploaded for reference file format**

