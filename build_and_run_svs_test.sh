#!/bin/bash
# Build and run the SVS serialization test

set -e

echo "=== Building SVS Serialization Test ==="

# Check if we're in the right directory
if [ ! -f "test_svs_serialization.cpp" ]; then
    echo "Error: test_svs_serialization.cpp not found"
    echo "Please run this script from the k-NN directory"
    exit 1
fi

# Set paths
FAISS_INCLUDE="jni/external/faiss"
FAISS_LIB="jni/build/faiss/faiss"
TEST_BINARY="test_svs_serialization"

echo "Compiling test..."
g++ -std=c++17 -g -o "$TEST_BINARY" test_svs_serialization.cpp \
    -I "$FAISS_INCLUDE" \
    -L "$FAISS_LIB" \
    -lfaiss_avx512_spr \
    -Wl,-rpath,"$FAISS_LIB" \
    -pthread

echo "✓ Compilation successful"
echo ""
echo "=== Running Test ==="
echo ""

./"$TEST_BINARY"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
else
    echo ""
    echo "✗ Some tests failed (exit code: $exit_code)"
fi

exit $exit_code
