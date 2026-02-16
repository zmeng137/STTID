#!/bin/bash

set -e  # Exit on error

echo "Building STTID project..."

# Build C++/CUDA STTID
echo "Building STTID..."
cd Cpp
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..

# Build Dense TT-ID
echo "Building Dense TT-ID..."
cd ttid
rm -rf build
mkdir build
cd build
cmake ..
make
cd ../..

echo "Build complete!"
echo "STTID executables: Cpp/build/"
echo "Dense TT-ID executables: ttid/build/"