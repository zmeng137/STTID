#!/bin/bash

set -e  # Exit on error

echo "Building STTID project..."

# Build C++/CUDA STTID
echo "Building STTID..."
cd src
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..
echo "STTID Build complete! Find executables in STTID/src/build/exe/ or STTID/exe/"

# Build Dense TT-ID
echo "Building Dense TT-ID..."
cd ttid
rm -rf build
mkdir build
cd build
cmake ..
make
cd ../..
echo "Dense TT-ID Build complete!"

echo "STTID executables: STTID/src/build/exe/ or STTID/exe/"
echo "Dense TT-ID executables: STTID/ttid/build/exe/ or STTID/exe/"