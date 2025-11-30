# Phone Sensor Fusion Application

A real-time sensor fusion application that displays your phone's rotation by streaming accelerometer, gyroscope, orientation, and magnetic field data over a network.

## 🚀 Quick Start (One Command)

To build the project, set up the environment, and launch everything:

```bash
./run_demo.sh
```

This script will:
1. **Check dependencies** and create a Python virtual environment (`venv`) if missing.
2. **Ask you to select an IP address** to bind the server to.
3. **Build the C++ project**.
4. **Start the server** and the **3D visualizer**.

Press `Ctrl+C` to stop everything cleanly.

## 🛠️ Setup & Requirements

### Prerequisites
- **Linux / WSL 2** (Ubuntu 20.04+ recommended)
- **CMake** 3.10+
- **G++** (C++17 support)
- **Python 3** (with `venv` support)
- **SensorFusion Library** (must be installed or in include path)

### Initial Setup
The `run_demo.sh` script handles most setup automatically. It uses `setup_env.sh` to:
- Check for required system tools (`cmake`, `make`, `g++`, `python3`).
- Create a Python virtual environment (`venv`).
- Install Python dependencies from `requirements.txt` (`pygame`, `PyOpenGL`).

To run setup manually:
```bash
./setup_env.sh
```

## 📱 Connection Guide

1. **Run the App**: Execute `./run_demo.sh`.
2. **Select IP**: The script will list available IP addresses.
   - If connecting from a phone on the same **Wi-Fi**, choose your LAN IP (e.g., `192.168.1.x`).
   - If using **WSL 2**, choose `0.0.0.0` (or the WSL IP `172.x.x.x`).

### WSL 2 Specific Setup
If you are running this inside WSL 2, you must allow the connection through the Windows Firewall.
1. Save the `setup_wsl2_firewall.ps1` script to your Windows Desktop (or access it from `\\wsl$\...`).
2. Right-click and **Run with PowerShell** (as Administrator).
3. Connect your phone to the **Windows IP** displayed by the script (e.g., `192.168.1.199`), not the internal WSL IP.

## 📱 Connection Guide
   - Open your sensor streaming app.
   - Set the **Target IP** to the one you selected.
   - Set the **Port** to `3400`.
   - Enable **Stream** and select sensors: **Accelerometer**, **Gyroscope**, **Magnetometer**, **Orientation**.
4. **Start Streaming**: Tap "Start" on your phone.

## Manual Operation

If you prefer to run components individually:

1. **Activate Environment**: `source venv/bin/activate`
2. **Build**:
   ```bash
   mkdir -p build && cd build
   cmake .. && make
   cd ..
   ```
3. **Run Server**: `./build/my_app --bind <YOUR_IP>`
4. **Run Visualizer**: `python3 phone_visualizer.py`

## Troubleshooting

- **"Missing dependencies"**: Run `./setup_env.sh` to verify and install missing tools.
- **"Bind failed"**: Ensure no other instance is running (`pkill -9 my_app`).
- **Visualizer error**: Ensure you are running in a graphical environment (or X server on Windows for WSL).
- **Inverted Rotation**: The visualizer expects standard Android coordinate systems. Some apps may differ.

## Project Structure

- `run_demo.sh`: Main entry point. Orchestrates build, setup, and execution.
- `setup_env.sh`: Checks dependencies and sets up Python venv.
- `get_ip.sh`: Helper to discover and select network interfaces.
- `main.cpp`: C++ Server & Sensor Fusion core.
- `phone_visualizer.py`: Python 3D visualization tool.
- `requirements.txt`: Python package list.

## License

See LICENSE file for details.
