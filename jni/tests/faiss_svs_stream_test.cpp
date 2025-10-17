// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.

#include "test_util.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <cstring>
#include <random>

#ifdef FAISS_ENABLE_SVS
#include <faiss/svs/IndexSVSFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>

/**
 * Simple IOWriter that writes to a vector buffer.
 * This directly implements faiss::IOWriter interface, avoiding the need for JNI mediators.
 * Similar to the approach used in the standalone test_svs_knn_exact.cpp
 */
class BufferIOWriter : public faiss::IOWriter {
private:
    std::vector<uint8_t>& buffer;
    
public:
    explicit BufferIOWriter(std::vector<uint8_t>& buf) : buffer(buf) {}
    
    size_t operator()(const void* ptr, size_t size, size_t nitems) override {
        const uint8_t* bytes = static_cast<const uint8_t*>(ptr);
        size_t total_bytes = size * nitems;
        buffer.insert(buffer.end(), bytes, bytes + total_bytes);
        return nitems;
    }
};

/**
 * Simple IOReader that reads from a vector buffer.
 * This directly implements faiss::IOReader interface, avoiding the need for JNI mediators.
 * Similar to the approach used in the standalone test_svs_knn_exact.cpp
 */
class BufferIOReader : public faiss::IOReader {
private:
    const std::vector<uint8_t>& buffer;
    size_t position;
    
public:
    explicit BufferIOReader(const std::vector<uint8_t>& buf) 
        : buffer(buf), position(0) {}
    
    size_t operator()(void* ptr, size_t size, size_t nitems) override {
        size_t total_bytes = size * nitems;
        size_t available = buffer.size() - position;
        size_t to_read = std::min(total_bytes, available);
        
        if (to_read > 0) {
            std::memcpy(ptr, buffer.data() + position, to_read);
            position += to_read;
        }
        
        // Return number of complete items read (not bytes)
        size_t items_read = to_read / size;
        return items_read;
    }
    
    void reset() { position = 0; }
    size_t getPosition() const { return position; }
};

/**
 * Test: SVS Flat index serialization and deserialization through streaming IOWriter/IOReader
 */
TEST(FaissSVSStreamTest, SVSFlat_SerializeDeserialize_BasicFlow) {
    // Create a small SVS Flat index
    int dimension = 128;
    int numVectors = 100;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> data(dimension * numVectors);
    for (auto& val : data) {
        val = dist(rng);
    }
    
    // Create and populate index
    faiss::IndexSVSFlat originalIndex(dimension, faiss::METRIC_L2);
    originalIndex.add(numVectors, data.data());
    
    ASSERT_EQ(originalIndex.ntotal, numVectors);
    ASSERT_EQ(originalIndex.d, dimension);
    
    // Serialize using BufferIOWriter
    std::vector<uint8_t> buffer;
    BufferIOWriter writer(buffer);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    }) << "Serialization should succeed";
    
    ASSERT_GT(buffer.size(), 0) << "Data should have been written";
    std::cout << "[TEST] Wrote " << buffer.size() << " bytes" << std::endl;
    
    // Deserialize using BufferIOReader
    BufferIOReader reader(buffer);
    
    faiss::Index* readIndex = nullptr;
    ASSERT_NO_THROW({
        readIndex = faiss::read_index(
            &reader,
            faiss::IO_FLAG_READ_ONLY 
            | faiss::IO_FLAG_PQ_SKIP_SDC_TABLE
            | faiss::IO_FLAG_SKIP_PRECOMPUTE_TABLE
        );
    }) << "Deserialization should succeed";
    
    ASSERT_NE(readIndex, nullptr) << "Read index should not be null";
    
    // Verify the index
    ASSERT_EQ(readIndex->ntotal, numVectors) << "Vector count mismatch";
    ASSERT_EQ(readIndex->d, dimension) << "Dimension mismatch";
    
    // Verify it's an SVS Flat index
    auto* svsIndex = dynamic_cast<faiss::IndexSVSFlat*>(readIndex);
    ASSERT_NE(svsIndex, nullptr) << "Index should be IndexSVSFlat";
    
    std::cout << "[TEST] ✓ Successfully serialized and deserialized SVS Flat index" << std::endl;
    std::cout << "[TEST]   Original: " << originalIndex.ntotal << " vectors, " 
              << originalIndex.d << " dimensions" << std::endl;
    std::cout << "[TEST]   Read:     " << readIndex->ntotal << " vectors, " 
              << readIndex->d << " dimensions" << std::endl;
    
    delete readIndex;
}

/**
 * Test: Multiple serialize/deserialize cycles
 */
TEST(FaissSVSStreamTest, SVSFlat_MultipleCycles) {
    int dimension = 64;
    int numVectors = 50;
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> data(dimension * numVectors);
    for (auto& val : data) {
        val = dist(rng);
    }
    
    faiss::IndexSVSFlat* currentIndex = new faiss::IndexSVSFlat(dimension, faiss::METRIC_L2);
    currentIndex->add(numVectors, data.data());
    
    // Perform 3 serialize/deserialize cycles
    for (int cycle = 0; cycle < 3; cycle++) {
        std::vector<uint8_t> buffer;
        BufferIOWriter writer(buffer);
        
        ASSERT_NO_THROW({
            faiss::write_index(currentIndex, &writer);
        }) << "Cycle " << cycle << " serialization failed";
        
        BufferIOReader reader(buffer);
        
        faiss::Index* newIndex = nullptr;
        ASSERT_NO_THROW({
            newIndex = faiss::read_index(&reader, faiss::IO_FLAG_READ_ONLY);
        }) << "Cycle " << cycle << " deserialization failed";
        
        ASSERT_EQ(newIndex->ntotal, numVectors) << "Cycle " << cycle << " vector count mismatch";
        ASSERT_EQ(newIndex->d, dimension) << "Cycle " << cycle << " dimension mismatch";
        
        delete currentIndex;
        currentIndex = dynamic_cast<faiss::IndexSVSFlat*>(newIndex);
        ASSERT_NE(currentIndex, nullptr) << "Cycle " << cycle << " index type mismatch";
    }
    
    std::cout << "[TEST] ✓ Successfully completed 3 serialize/deserialize cycles" << std::endl;
    delete currentIndex;
}

/**
 * Test: Large index serialization
 */
TEST(FaissSVSStreamTest, SVSFlat_LargeIndex) {
    int dimension = 256;
    int numVectors = 1000;
    
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> data(dimension * numVectors);
    for (auto& val : data) {
        val = dist(rng);
    }
    
    faiss::IndexSVSFlat originalIndex(dimension, faiss::METRIC_L2);
    originalIndex.add(numVectors, data.data());
    
    std::vector<uint8_t> buffer;
    BufferIOWriter writer(buffer);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    });
    
    std::cout << "[TEST] Large index serialized: " << buffer.size() 
              << " bytes (" << (buffer.size() / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    BufferIOReader reader(buffer);
    
    faiss::Index* readIndex = nullptr;
    ASSERT_NO_THROW({
        readIndex = faiss::read_index(&reader, faiss::IO_FLAG_READ_ONLY);
    });
    
    ASSERT_EQ(readIndex->ntotal, numVectors);
    ASSERT_EQ(readIndex->d, dimension);
    
    std::cout << "[TEST] ✓ Large index deserialized successfully" << std::endl;
    delete readIndex;
}

/**
 * Test: Empty index serialization
 */
TEST(FaissSVSStreamTest, SVSFlat_EmptyIndex) {
    int dimension = 128;
    
    faiss::IndexSVSFlat originalIndex(dimension, faiss::METRIC_L2);
    // Don't add any vectors
    
    ASSERT_EQ(originalIndex.ntotal, 0);
    
    std::vector<uint8_t> buffer;
    BufferIOWriter writer(buffer);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    });
    
    ASSERT_GT(buffer.size(), 0) << "Even empty index should write metadata";
    
    BufferIOReader reader(buffer);
    
    faiss::Index* readIndex = nullptr;
    ASSERT_NO_THROW({
        readIndex = faiss::read_index(&reader, faiss::IO_FLAG_READ_ONLY);
    });
    
    ASSERT_EQ(readIndex->ntotal, 0) << "Empty index should have 0 vectors";
    ASSERT_EQ(readIndex->d, dimension);
    
    std::cout << "[TEST] ✓ Empty index serialized and deserialized successfully" << std::endl;
    delete readIndex;
}

/**
 * Test: Verify search works after deserialization
 */
TEST(FaissSVSStreamTest, SVSFlat_SearchAfterDeserialization) {
    int dimension = 128;
    int numVectors = 100;
    int k = 5;
    
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> data(dimension * numVectors);
    for (auto& val : data) {
        val = dist(rng);
    }
    
    faiss::IndexSVSFlat originalIndex(dimension, faiss::METRIC_L2);
    originalIndex.add(numVectors, data.data());
    
    // Perform a search on original index
    std::vector<float> query(dimension);
    for (auto& val : query) {
        val = dist(rng);
    }
    
    std::vector<faiss::idx_t> originalLabels(k);
    std::vector<float> originalDistances(k);
    originalIndex.search(1, query.data(), k, originalDistances.data(), originalLabels.data());
    
    // Serialize and deserialize
    std::vector<uint8_t> buffer;
    BufferIOWriter writer(buffer);
    faiss::write_index(&originalIndex, &writer);
    
    BufferIOReader reader(buffer);
    faiss::Index* readIndex = faiss::read_index(&reader, faiss::IO_FLAG_READ_ONLY);
    
    // Perform the same search on deserialized index
    std::vector<faiss::idx_t> readLabels(k);
    std::vector<float> readDistances(k);
    readIndex->search(1, query.data(), k, readDistances.data(), readLabels.data());
    
    // Results should be identical
    for (int i = 0; i < k; i++) {
        ASSERT_EQ(originalLabels[i], readLabels[i]) 
            << "Label mismatch at position " << i;
        ASSERT_NEAR(originalDistances[i], readDistances[i], 1e-5) 
            << "Distance mismatch at position " << i;
    }
    
    std::cout << "[TEST] ✓ Search results match after deserialization" << std::endl;
    delete readIndex;
}

#endif // FAISS_ENABLE_SVS
