
import aubio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json
import csv
from datetime import datetime
import os

# Define low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Fit a polynomial of desired degree (e.g., degree 5)
def iterative_smooth_and_filter(times, frequencies, degree=5, threshold=100, iterations=3):
    # Convert inputs to numpy arrays
    times = np.array(times)
    frequencies = np.array(frequencies)
    
    # Step 1: Initial fit to the data
    current_times = times
    current_frequencies = frequencies.copy()
    
    for i in range(iterations):
        # Fit polynomial to the current frequencies
        coefficients = np.polyfit(current_times, current_frequencies, degree)
        polynomial = np.poly1d(coefficients)
        
        # Calculate the fitted values (y-values of the polynomial) at the given times
        fitted_values = polynomial(current_times)
        
        # Step 2: Adjust fitting curve points where deviation > threshold
        deviations = np.abs(frequencies - fitted_values)
        current_frequencies = np.where(deviations > threshold, fitted_values, frequencies)
    
    # Generate smooth values for plotting the final curve after iterations
    smooth_times = np.linspace(times.min(), times.max(), 500)
    smooth_frequencies = polynomial(smooth_times)

    return smooth_times, smooth_frequencies


def get_note(frequency,freq_to_note):
    for freq_range, note in freq_to_note.items():
        low, high = map(float, freq_range.split('-'))
        if low <= frequency <= high:
            return note
    return None

# Before creating the plot, let's create the note regions
def add_note_regions(plt, freq_to_note, times):
    # Define the note range we want (G2 to C4)
    min_freq = 98  # G2
    max_freq = 523.25  # C5
    
    # Convert freq_to_note ranges to visualization
    for freq_range, note in freq_to_note.items():
        low, high = map(float, freq_range.split('-'))
        # Only add regions within our desired range
        if low >= min_freq and high <= max_freq:
            plt.axhspan(low, high, color='gray', alpha=0.1)
            # Add note label on the right side
            plt.text(max(times) + 0.1, (low + high)/2, note, 
                    verticalalignment='center', fontsize=8)


