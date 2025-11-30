#!/usr/bin/env python3
"""
Generate WAV files for testing the localization system.
Creates 4 WAV files with simulated TDOA based on a known source position.
"""

import numpy as np
import argparse
import struct

def write_wav(filename, data, sample_rate=44100):
    """
    Write a simple mono WAV file.
    
    Args:
        filename: Output filename
        data: Audio samples in range [-1, 1]
        sample_rate: Sample rate in Hz
    """
    # Scale to int16 range
    data_int16 = (data * 32767).astype(np.int16)
    
    num_samples = len(data_int16)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    
    with open(filename, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # fmt chunk size
        f.write(struct.pack('<H', 1))   # audio format (1 = PCM)
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(data_int16.tobytes())

def generate_audio_with_tdoa(source_pos, mic_pos, num_samples=1024, sample_rate=44100, freq=200, noise_level=0.01):
    """Generate audio signal with appropriate delay."""
    distance = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
    sound_speed = 343.0
    delay_seconds = distance / sound_speed
    
    t = np.arange(num_samples) / sample_rate
    amplitude = 0.5 / (1.0 + distance)
    signal = amplitude * np.sin(2 * np.pi * freq * (t - delay_seconds))
    noise = np.random.normal(0, noise_level, num_samples)
    
    return signal + noise

def main():
    parser = argparse.ArgumentParser(description='Generate test WAV files for localization')
    parser.add_argument('--source-x', type=float, default=0.15, help='Source X position (m)')
    parser.add_argument('--source-y', type=float, default=0.10, help='Source Y position (m)')
    parser.add_argument('--source-z', type=float, default=0.05, help='Source Z position (m)')
    parser.add_argument('--samples', type=int, default=1024, help='Number of samples')
    parser.add_argument('--freq', type=float, default=200, help='Frequency (Hz)')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise level')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    source_pos = [args.source_x, args.source_y, args.source_z]
    
    mic_positions = [
        [0.0, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.20, 0.20, 0.0],
        [0.0, 0.20, 0.0]
    ]
    
    print(f"Generating WAV files for source at ({source_pos[0]}, {source_pos[1]}, {source_pos[2]})")
    print(f"Sample rate: 44100 Hz, {args.samples} samples")
    print()
    
    for i, mic_pos in enumerate(mic_positions, 1):
        audio = generate_audio_with_tdoa(
            source_pos, mic_pos,
            num_samples=args.samples,
            freq=args.freq,
            noise_level=args.noise
        )
        
        filename = f"{args.output_dir}/mic{i}.wav"
        write_wav(filename, audio)
        print(f"Created {filename}")
    
    print()
    print("Files created successfully!")
    print()
    print("To run localization:")
    print(f"  ./localize_from_files {args.output_dir}/mic1.wav {args.output_dir}/mic2.wav {args.output_dir}/mic3.wav {args.output_dir}/mic4.wav")

if __name__ == '__main__':
    main()
