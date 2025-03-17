## ❗️Update from the pitch_detection.py: 
## ❗️Save the note sequence to a JSON file rather than a CSV file.

import aubio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json
import csv
from datetime import datetime
import os
from audio_processing.functions.pitch_processing import add_note_regions, get_note, iterative_smooth_and_filter, apply_lowpass_filter

def process_audio_file(audio_file_path, save_plot=True, show_plot=False, save_json=True):
    """
    Process an audio file to detect pitch and extract note sequences.
    
    Args:
        audio_file_path (str): Path to the audio file
        save_plot (bool): Whether to save the plot as an image
        show_plot (bool): Whether to display the plot
        save_json (bool): Whether to save the note sequence to a JSON file
        
    Returns:
        dict: Contains note_sequence, times, pitches, and output_path (if saved)
    """
    # Set parameters
    samplerate = 44100  # Set to 0 to use the file's original sample rate
    hop_size = 512      # Number of frames between each analysis

    # Initialize source and get audio data
    src = aubio.source(audio_file_path, samplerate, hop_size)
    samplerate = src.samplerate

    # Read the entire audio file into a numpy array
    audio_data = []
    while True:
        samples, read = src()
        audio_data.extend(samples)
        if read < hop_size:
            break
    audio_data = np.array(audio_data)

    # Apply low-pass filter (cutoff at 800 Hz)
    audio_data = np.array(audio_data)
    filtered_audio = apply_lowpass_filter(audio_data, cutoff=800, fs=samplerate)

    # Reinitialize source with noise-reduced audio
    src = aubio.source(audio_file_path, samplerate, hop_size)  # Reset source position
    pitch_o = aubio.pitch("default", 2048, hop_size, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_silence(-40)
    pitch_o.set_tolerance(0.6)

    # Process the audio file
    pitches = []
    times = []  # Add time tracking
    total_frames = 0
    while True:
        samples, read = src()
        pitch = pitch_o(samples)[0]
        if 98 <= pitch <= 440:  # Only consider pitches between G2 and C4
            pitches.append(pitch)
            times.append(total_frames / float(samplerate))
        total_frames += read
        if read < hop_size:
            break

    # Smooth and filter the pitches
    smooth_times, smooth_frequencies = iterative_smooth_and_filter(times, pitches)

    # Load the frequency to note mapping
    with open('./audio_processing/freq2note.json', 'r') as f:
        freq_to_note = json.load(f)

    # Create note sequence data
    note_sequence = []
    for time, freq in zip(times, pitches):
        note = get_note(freq, freq_to_note)
        if note:
            note_sequence.append({
                'time': float(round(time, 2)),  # Convert to Python float
                'note': note,
                'frequency': float(round(freq, 1))  # Convert to Python float
            })

    # Create and save plot if requested
    if save_plot or show_plot:
        plt.figure(figsize=(12, 6))
        
        # Add the note regions first (so they're in the background)
        add_note_regions(plt, freq_to_note, times)
        
        # Plot frequencies
        plt.plot(times, pitches, 'b.')
        
        # Add note labels where they change
        current_note = None
        note_positions = []
        note_labels = []
        note_times = []
        
        for time, freq in zip(times, pitches):
            note = get_note(freq, freq_to_note)
            if note != current_note:
                current_note = note
                if note:
                    note_positions.append(freq)
                    note_labels.append(note)
                    note_times.append(time)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Pitch Frequencies Over Time')
        plt.grid(True)
        
        # Set x-axis ticks to show 1-second intervals
        max_time = max(times) if times else 0
        plt.xticks(range(0, int(max_time) + 1, 1))
        
        # Adjust the plot limits to show note labels
        plt.margins(x=0.05)  # Add 5% padding to the right for note labels
        
        # Set y-axis limits to show only C2 to C4 range
        plt.ylim(98, 523.25)  # From G2 (98 Hz) to C5 (523.25 Hz)
        
        if save_plot:
            # Create data directory if it doesn't exist
            data_dir = 'analysis/plot'
            os.makedirs(data_dir, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(data_dir, f'pitch_plot_{current_date}.png')
            plt.savefig(plot_filename)
            # print(f"Plot has been saved to '{plot_filename}'")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    # Save note sequence to JSON if requested
    output_path = None
    if save_json:
        # Create data directory if it doesn't exist
        data_dir = 'analysis/pitch'
        os.makedirs(data_dir, exist_ok=True)
        
        # Save note sequence to JSON with date-based filename
        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = os.path.join(data_dir, f'note_sequence_{current_date}.json')
        
        # Write to JSON file
        with open(json_filename, 'w') as jsonfile:
            json.dump(note_sequence, jsonfile, indent=4)
        
        # print(f"Note sequence has been saved to '{json_filename}'")
        output_path = json_filename

    return {
        'note_sequence': note_sequence,
        'times': times,
        'pitches': pitches,
        'output_path': output_path
    }

