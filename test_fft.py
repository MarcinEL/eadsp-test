#!/usr/bin/env python3
"""
test_fft.py

This script tests a student’s FFT function. The student’s FFT implementation should be
provided in a Python script (specified via the command-line argument --student-script) and must define a
function with the signature:

    compute_fft(signal, fs)

which returns a tuple: (freq, magnitude, phase).

The script generates a composite signal, computes its FFT using the student's function,
plots the magnitude and phase spectra, and performs an automated check for expected frequency peaks
(50 Hz and 120 Hz).
"""

import argparse
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_student_function(filepath):
    """
    Load a student's function from a given script file.

    Parameters:
        filepath (str): Path to the student's Python script.
    
    Returns:
        function: The student's compute_fft function.
    
    Raises:
        ImportError or AttributeError if the module cannot be loaded or does not define compute_fft.
    """
    spec = importlib.util.spec_from_file_location("student_module", filepath)
    if spec is None:
        raise ImportError(f"Could not load module spec from file: {filepath}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing the module {filepath}: {e}") from e
    
    if not hasattr(module, "compute_fft"):
        raise AttributeError("The provided script does not define a function named 'compute_fft'.")
    
    return getattr(module, "compute_fft")

def test_student_fft(student_fft_func):
    """
    Test the student's FFT function using a composite signal.
    
    Parameters:
        student_fft_func (function): The student's FFT function.
    """
    # Test signal parameters
    fs = 1000  # Sampling frequency (Hz)
    t = np.arange(0, 1, 1/fs)
    amps = [1.0, 0.5, 0.125, 0.33]
    f = [120, 50, 124, 483]
    signal = np.zeros_like(t)
    # Create a composite signal by summing sinusoids
    for freq, amp in zip(f, amps):
        signal += amp * np.sin(2 * np.pi * freq * t)
        

    # Execute the student's FFT function.
    try:
        freq, magnitude, phase = student_fft_func(signal, fs)
    except Exception as e:
        print("Error when calling the student's compute_fft function:", e)
        sys.exit(1)

    # Plot the FFT results.
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(freq, magnitude, marker='o')
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(freq, phase, marker='o')
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Automated check: report the top 5 frequency peaks and verify that peaks near 50 Hz and 120 Hz exist.
    peak_indices = np.argsort(magnitude)[-5:]
    peak_freqs = np.sort(np.abs(np.array(freq)[peak_indices]))
    print("Detected peak frequencies (Hz):", peak_freqs)
    
    if any(np.isclose(peak_freqs, 50, atol=1)) and any(np.isclose(peak_freqs, 120, atol=1)):
        print("Test Passed: Expected peaks " + str(f) + " were detected.")
    else:
        print("Test Failed: Expected frequency peaks not detected.")

def main():
    parser = argparse.ArgumentParser(
        description="Test a student's FFT implementation from a provided script file."
    )
    parser.add_argument(
        '--student-script',
        type=str,
        required=True,
        help='Path to the student script containing the compute_fft function.'
    )
    args = parser.parse_args()

    # Load the student's compute_fft function from the given script file.
    try:
        student_fft_func = load_student_function(args.student_script)
    except Exception as e:
        print("Error loading the student's function:", e)
        sys.exit(1)
    
    print("Successfully loaded the student's compute_fft function from:", args.student_script)
    
    # Run the FFT test.
    test_student_fft(student_fft_func)

if __name__ == '__main__':
    main()
