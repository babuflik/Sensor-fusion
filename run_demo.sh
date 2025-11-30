#!/bin/bash

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    pkill -P $$ # Kill child processes of this script
    pkill -9 my_app 2>/dev/null
    pkill -f "python3 phone_visualizer.py" 2>/dev/null
    echo "✅ Cleanup complete."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# 0. Check Environment & Dependencies
if [ ! -d "venv" ]; then
    echo "⚠️  Python virtual environment not found. Running setup..."
    ./setup_env.sh
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed."
        exit 1
    fi
fi

# Activate venv
source venv/bin/activate

# Ensure dependencies are up to date
echo "📦 Checking Python dependencies..."
pip install -r requirements.txt > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    exit 1
fi

# 1. IP Selection
# Source the get_ip.sh script to get SENSOR_FUSION_IP
# We run it in a subshell to capture output if needed, but here we just want the var
# Actually, let's just run it interactively if no arg provided
if [ -z "$1" ]; then
    source ./get_ip.sh
    BIND_IP=$SENSOR_FUSION_IP
else
    BIND_IP=$1
fi

# 2. Cleanup previous instances
echo "🧹 Cleaning up previous instances..."
pkill -9 my_app 2>/dev/null
pkill -f "python3 phone_visualizer.py" 2>/dev/null

# 2. Build the project
echo "🔨 Building project..."
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake .. > /dev/null
make -j$(nproc) > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Build failed."
    exit 1
fi
cd ..
echo "✅ Build successful."

# 3. Start C++ Server
echo "🚀 Starting Sensor Fusion Server (Bind: $BIND_IP)..."
./build/my_app --bind $BIND_IP > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to initialize
sleep 1

# 4. Start Visualizer
echo "📱 Starting 3D Visualizer..."
python3 phone_visualizer.py &
VISUALIZER_PID=$!

echo "==================================================="
echo "   Sensor Fusion Demo Running"
echo "   Server PID: $SERVER_PID"
echo "   Visualizer PID: $VISUALIZER_PID"
echo "   Press Ctrl+C to stop everything"
echo "==================================================="

# Keep script running to maintain trap
wait
