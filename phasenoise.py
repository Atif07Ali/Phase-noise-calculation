import numpy as np
from scipy.signal import windows
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import sys

def preprocess_ngspice_file(file_path):
    numeric_values = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                numeric_values.append(float(line.strip()))
            except ValueError:
                continue

    raw = np.array(numeric_values)
    if len(raw) % 2 != 0:
        raw = raw[:-1]

    N = len(raw) // 2
    t = raw[:N]
    v = raw[N:]
    dt = t[1] - t[0]
    v_centered = v - np.mean(v)

    window = windows.hann(N)
    v_windowed = v_centered * window
    rms_gain = np.sqrt(np.sum(window**2) / N)

    return t, v_windowed, dt, N, rms_gain

def compute_fft_psd(v_windowed, dt, N, rms_gain):
    fft_result = fft(v_windowed)
    freq = fftfreq(N, dt)[:N // 2]
    df = freq[1] - freq[0]
    magnitude = (2 / (N * rms_gain)) * np.abs(fft_result[:N // 2])
    psd = (magnitude ** 2) / df
    return freq, magnitude, psd, df

def annotate_and_compute_phase_noise(freq, magnitude, psd, df, offset_hz=1e6):
    print(f"📐 FFT resolution (df): {df:.1f} Hz")

    peak_idx = np.argmax(magnitude)
    peak_freq = freq[peak_idx]
    peak_val = magnitude[peak_idx]

    carrier_power = np.sum(psd[max(peak_idx-1, 0):peak_idx+2]) * df
    print(f"🔋 Carrier Power ≈ {carrier_power:.2e} V²")

    offset_bins = int(round(offset_hz / df))
    offset_idx_upper = peak_idx + offset_bins
    offset_idx_lower = peak_idx - offset_bins

    if offset_idx_upper >= len(freq) or offset_idx_lower < 0:
        raise ValueError("Offset index out of range. Increase simulation time or FFT length.")

    psd_upper = psd[offset_idx_upper]
    psd_lower = psd[offset_idx_lower]

    if psd_upper > psd_lower:
        offset_idx = offset_idx_upper
        offset_label = "+1 MHz"
    else:
        offset_idx = offset_idx_lower
        offset_label = "-1 MHz"

    offset_freq = freq[offset_idx]
    offset_val = magnitude[offset_idx]
    offset_psd_val = psd[offset_idx]

    phase_noise = 10 * np.log10(offset_psd_val / carrier_power)

    print(f"🔺 Peak: {peak_freq/1e9:.6f} GHz | Mag: {peak_val:.2e} V | PSD: {10*np.log10(psd[peak_idx]):.2f} dBV²/Hz")
    print(f"📍 Offset @ {offset_freq/1e6:.2f} MHz ({offset_label}) | PSD: {10*np.log10(offset_psd_val):.2f} dBV²/Hz")
    print(f"📉 Phase Noise @ 1 MHz offset: {phase_noise:.2f} dBc/Hz")

    return peak_freq, peak_val, offset_freq, offset_val, offset_psd_val, carrier_power

def plot_spectrum(freq, magnitude, psd, peak_f, peak_v, offset_f, offset_v):
    plt.figure(figsize=(10, 5))
    plt.plot(freq / 1e9, magnitude, label='FFT Magnitude')
    plt.plot(peak_f / 1e9, peak_v, 'ro', label='Peak')
    plt.plot(offset_f / 1e9, offset_v, 'go', label='1 MHz Offset')
    plt.annotate(f"Peak\n{peak_f/1e9:.3f} GHz\n{peak_v:.2e} V",
                 (peak_f / 1e9, peak_v), textcoords="offset points", xytext=(10, 10), ha='left')
    plt.annotate(f"Offset\n{offset_f/1e9:.3f} GHz\n{offset_v:.2e} V",
                 (offset_f / 1e9, offset_v), textcoords="offset points", xytext=(10, -30), ha='left')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Amplitude (V)")
    plt.title("FFT Magnitude Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(freq / 1e9, 10 * np.log10(psd), label='PSD (dBV²/Hz)')
    plt.plot(peak_f / 1e9, 10*np.log10(psd[np.argmax(freq >= peak_f)]), 'ro', label='Peak')
    plt.plot(offset_f / 1e9, 10*np.log10(psd[np.argmax(freq >= offset_f)]), 'go', label='1 MHz Offset')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (dBV²/Hz)")
    plt.title("Power Spectral Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- CLI ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ngspice_phase_noise_corrected.py input_file.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    t, v_windowed, dt, N, rms_gain = preprocess_ngspice_file(input_file)
    freq, magnitude, psd, df = compute_fft_psd(v_windowed, dt, N, rms_gain)
    peak_f, peak_v, offset_f, offset_v, offset_psd, p_carrier = annotate_and_compute_phase_noise(freq, magnitude, psd, df)
    plot_spectrum(freq, magnitude, psd, peak_f, peak_v, offset_f, offset_v)
