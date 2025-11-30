#!/usr/bin/env python3
"""
Generate sample audio files for testing 3-mic multi-recording localization.
Creates multiple recordings with 3 mics each from a stationary source.
"""

import numpy as np
import argparse
import os

def generate_audio_with_tdoa(source_pos, mic_pos, num_samples=8192, sample_rate=44100, freq=200, noise_level=0.01):
    """Generate audio signal with appropriate delay based on source and mic position."""
    distance = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
    sound_speed = 343.0
    delay_seconds = distance / sound_speed
    t = np.arange(num_samples) / sample_rate
    amplitude = 0.5 / (1.0 + distance)
    signal = amplitude * np.sin(2 * np.pi * freq * (t - delay_seconds))
    noise = np.random.normal(0, noise_level, num_samples)
    return signal + noise

def main():
    parser = argparse.ArgumentParser(description='Generate test audio files for 3-mic multi-recording localization')
    parser.add_argument('--source-x', type=float, default=0.35, help='Source X position (m)')
    parser.add_argument('--source-y', type=float, default=0.25, help='Source Y position (m)')
    parser.add_argument('--source-z', type=float, default=0.60, help='Source Z position (m)')
    parser.add_argument('--samples', type=int, default=8192, help='Number of samples')
    parser.add_argument('--freq', type=float, default=200, help='Frequency (Hz)')
    parser.add_argument('--noise', type=float, default=0.005, help='Noise level')
    parser.add_argument('--recordings', type=int, default=2, help='Number of recordings')
    parser.add_argument('--output-dir', type=str, default='test_3mic', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    source_pos = [args.source_x, args.source_y, args.source_z]
    
    recording_configs = [
        {'name': 'ground_level', 'mics': [[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.15, 0.26, 0.0]]},
        {'name': 'elevated', 'mics': [[0.0, 0.0, 0.8], [0.30, 0.0, 0.8], [0.15, 0.26, 0.8]]},
        {'name': 'different_position', 'mics': [[0.50, 0.50, 0.40], [0.80, 0.50, 0.40], [0.65, 0.76, 0.40]]}
    ]
    
    print(f"=== Generating 3-Mic Multi-Recording Test Data ===")
    print(f"Source position: ({source_pos[0]}, {source_pos[1]}, {source_pos[2]}) m")
    print(f"Number of recordings: {min(args.recordings, len(recording_configs))}")
    print(f"Samples per file: {args.samples}, Frequency: {args.freq} Hz, Noise: {args.noise}\n")
    
    command_args = []
    
    for rec_idx in range(min(args.recordings, len(recording_configs))):
        config = recording_configs[rec_idx]
        print(f"Recording {rec_idx + 1} ({config['name']}):")
        
        rec_args = []
        for mic_idx, mic_pos in enumerate(config['mics'], 1):
            audio = generate_audio_with_tdoa(source_pos, mic_pos, num_samples=args.samples, 
                                            sample_rate=44100, freq=args.freq, noise_level=args.noise)
            filename = f"rec{rec_idx+1}_mic{mic_idx}.csv"
            filepath = os.path.join(args.output_dir, filename)
            np.savetxt(filepath, audio, fmt='%.8f')
            
            print(f"  M{mic_idx}: {filepath} at ({mic_pos[0]:.2f}, {mic_pos[1]:.2f}, {mic_pos[2]:.2f}) m")
            rec_args.extend([filename, f"{mic_pos[0]},{mic_pos[1]},{mic_pos[2]}"])
        
        command_args.extend(rec_args)
        print()
    
    print("Files created successfully!\n")
    print("=" * 70)
    print("To run 3-mic multi-recording localization:")
    print("=" * 70)
    print(f"cd {args.output_dir}")
    print("../build/localize_multi_recording \\")
    
    for i in range(0, len(command_args), 6):
        line = "  " + " ".join(command_args[i:i+6])
        if i + 6 < len(command_args):
            line += " \\"
        print(line)
    
    print(f"\nExpected result: Position close to ({source_pos[0]:.2f}, {source_pos[1]:.2f}, {source_pos[2]:.2f}) m")

if __name__ == '__main__':
    main()
