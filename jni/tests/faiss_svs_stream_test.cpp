// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.

#include "faiss_stream_support.h"
#include "test_util.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <cstring>
#include <random>

#ifdef FAISS_ENABLE_SVS
#include <faiss/svs/IndexSVSFlat.h>
#include <faiss/index_io.h>

using knn_jni::stream::FaissOpenSearchIOReader;
using knn_jni::stream::FaissOpenSearchIOWriter;
using knn_jni::stream::NativeEngineIndexInputMediator;
using knn_jni::stream::NativeEngineIndexOutputMediator;

/**
 * Mock IOWriter that writes to a vector buffer (simulates Lucene IndexOutput)
 */
class MockIOWriterToBuffer : public NativeEngineIndexOutputMediator {
private:
    std::vector<uint8_t> buffer;
    
public:
    void writeBytes(const uint8_t* bytes, int32_t length) override {
        buffer.insert(buffer.end(), bytes, bytes + length);
    }
    
    void flush() override {
        // No-op for testing
    }
    
    const std::vector<uint8_t>& getBuffer() const { return buffer; }
    size_t size() const { return buffer.size(); }
    void clear() { buffer.clear(); }
};

/**
 * Mock IOReader that reads from a vector buffer (simulates Lucene IndexInput)
 */
class MockIOReaderFromBuffer : public NativeEngineIndexInputMediator {
private:
    const std::vector<uint8_t>& buffer;
    size_t position;
    std::vector<uint8_t> tempBuffer;
    
public:
    explicit MockIOReaderFromBuffer(const std::vector<uint8_t>& buf) 
        : buffer(buf), position(0), tempBuffer(1024 * 1024) {}  // 1MB temp buffer
    
    void copyBytes(int32_t length, uint8_t* destination) override {
        if (position + length > buffer.size()) {
            throw std::runtime_error("Read beyond buffer size");
        }
        std::memcpy(destination, buffer.data() + position, length);
        position += length;
    }
    
    int64_t remainingBytes() override {
        return buffer.size() - position;
    }
    
    void reset() { position = 0; }
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
    
    // Serialize using FaissOpenSearchIOWriter
    MockIOWriterToBuffer writerMediator;
    FaissOpenSearchIOWriter writer(&writerMediator);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    }) << "Serialization should succeed";
    
    ASSERT_GT(writerMediator.size(), 0) << "Data should have been written";
    std::cout << "[TEST] Wrote " << writerMediator.size() << " bytes" << std::endl;
    
    // Deserialize using FaissOpenSearchIOReader
    MockIOReaderFromBuffer readerMediator(writerMediator.getBuffer());
    FaissOpenSearchIOReader reader(&readerMediator);
    
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
        MockIOWriterToBuffer writerMediator;
        FaissOpenSearchIOWriter writer(&writerMediator);
        
        ASSERT_NO_THROW({
            faiss::write_index(currentIndex, &writer);
        }) << "Cycle " << cycle << " serialization failed";
        
        MockIOReaderFromBuffer readerMediator(writerMediator.getBuffer());
        FaissOpenSearchIOReader reader(&readerMediator);
        
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
    
    MockIOWriterToBuffer writerMediator;
    FaissOpenSearchIOWriter writer(&writerMediator);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    });
    
    std::cout << "[TEST] Large index serialized: " << writerMediator.size() 
              << " bytes (" << (writerMediator.size() / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    MockIOReaderFromBuffer readerMediator(writerMediator.getBuffer());
    FaissOpenSearchIOReader reader(&readerMediator);
    
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
    
    MockIOWriterToBuffer writerMediator;
    FaissOpenSearchIOWriter writer(&writerMediator);
    
    ASSERT_NO_THROW({
        faiss::write_index(&originalIndex, &writer);
    });
    
    ASSERT_GT(writerMediator.size(), 0) << "Even empty index should write metadata";
    
    MockIOReaderFromBuffer readerMediator(writerMediator.getBuffer());
    FaissOpenSearchIOReader reader(&readerMediator);
    
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
    MockIOWriterToBuffer writerMediator;
    FaissOpenSearchIOWriter writer(&writerMediator);
    faiss::write_index(&originalIndex, &writer);
    
    MockIOReaderFromBuffer readerMediator(writerMediator.getBuffer());
    FaissOpenSearchIOReader reader(&readerMediator);
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
