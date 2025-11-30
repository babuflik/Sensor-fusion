#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Checking Dependencies...${NC}"

# 1. Check System Tools
echo -n "Checking for CMake... "
if command -v cmake >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}MISSING${NC}"
    echo "Please install cmake (e.g., sudo apt install cmake)"
    exit 1
fi

echo -n "Checking for Make... "
if command -v make >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}MISSING${NC}"
    echo "Please install make (e.g., sudo apt install build-essential)"
    exit 1
fi

echo -n "Checking for G++... "
if command -v g++ >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}MISSING${NC}"
    echo "Please install g++ (e.g., sudo apt install g++)"
    exit 1
fi

echo -n "Checking for Python 3... "
if command -v python3 >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}MISSING${NC}"
    echo "Please install python3"
    exit 1
fi

# 2. Check SensorFusion Library
echo -n "Checking for SensorFusion library... "
# We'll check if we can find the header or library file in common locations
if [ -f "/usr/local/include/SensorFusion/SensorFusion.h" ] || \
   [ -f "/usr/include/SensorFusion/SensorFusion.h" ] || \
   [ -f "SensorFusion/SensorFusion.h" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "SensorFusion headers not found in standard paths."
    echo "Ensure the SensorFusion library is installed or present in the project directory."
    echo "If using a local copy, make sure CMakeLists.txt is configured correctly."
    # We don't exit here because it might be a local subproject not yet built
fi

# 3. Python Environment Setup
echo -e "\n${YELLOW}🐍 Setting up Python Environment...${NC}"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create venv.${NC}"
        echo "Try installing venv: sudo apt install python3-venv"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install Python dependencies.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies checked and environment ready.${NC}"
