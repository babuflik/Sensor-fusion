#!/usr/bin/env python3
"""
Generate sample audio files for testing the localization system.
Creates 4 CSV files with simulated TDOA based on a known source position.
"""

import numpy as np
import argparse

def generate_audio_with_tdoa(source_pos, mic_pos, num_samples=1024, sample_rate=44100, freq=200, noise_level=0.01):
    """
    Generate audio signal with appropriate delay based on source and mic position.
    
    Args:
        source_pos: [x, y, z] position of sound source
        mic_pos: [x, y, z] position of microphone
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        freq: Frequency of humming sound in Hz
        noise_level: Standard deviation of Gaussian noise
    
    Returns:
        Array of audio samples
    """
    # Calculate distance
    distance = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
    
    # Calculate time delay (speed of sound = 343 m/s)
    sound_speed = 343.0
    delay_seconds = distance / sound_speed
    
    # Generate time array
    t = np.arange(num_samples) / sample_rate
    
    # Generate signal with delay and 1/r attenuation
    amplitude = 0.5 / (1.0 + distance)  # Simple attenuation model
    signal = amplitude * np.sin(2 * np.pi * freq * (t - delay_seconds))
    
    # Add noise
    noise = np.random.normal(0, noise_level, num_samples)
    
    return signal + noise

def main():
    parser = argparse.ArgumentParser(description='Generate test audio files for localization')
    parser.add_argument('--source-x', type=float, default=0.15, help='Source X position (m)')
    parser.add_argument('--source-y', type=float, default=0.10, help='Source Y position (m)')
    parser.add_argument('--source-z', type=float, default=0.05, help='Source Z position (m)')
    parser.add_argument('--samples', type=int, default=1024, help='Number of samples')
    parser.add_argument('--freq', type=float, default=200, help='Frequency (Hz)')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise level')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    source_pos = [args.source_x, args.source_y, args.source_z]
    
    # Microphone positions (square array, 20cm spacing)
    mic_positions = [
        [0.0, 0.0, 0.0],      # M1
        [0.20, 0.0, 0.0],     # M2
        [0.20, 0.20, 0.0],    # M3
        [0.0, 0.20, 0.0]      # M4
    ]
    
    print(f"Generating audio files for source at ({source_pos[0]}, {source_pos[1]}, {source_pos[2]})")
    print(f"Microphone array: 4 mics in square configuration (0.2m spacing)")
    print()
    
    for i, mic_pos in enumerate(mic_positions, 1):
        # Generate audio
        audio = generate_audio_with_tdoa(
            source_pos, mic_pos, 
            num_samples=args.samples,
            freq=args.freq,
            noise_level=args.noise
        )
        
        # Save as CSV
        filename = f"{args.output_dir}/mic{i}.csv"
        np.savetxt(filename, audio, fmt='%.8f')
        print(f"Created {filename} ({len(audio)} samples)")
    
    print()
    print("Files created successfully!")
    print()
    print("To run localization:")
    print(f"  ./localize_from_files {args.output_dir}/mic1.csv {args.output_dir}/mic2.csv {args.output_dir}/mic3.csv {args.output_dir}/mic4.csv")

if __name__ == '__main__':
    main()
