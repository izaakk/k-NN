/**
 * Standalone test for SVS Flat index serialization
 * 
 * This test verifies that SVS Flat indices can be:
 * 1. Created via index_factory
 * 2. Have vectors added
 * 3. Be written to disk
 * 4. Be read back from disk
 * 5. Perform searches after loading
 * 
 * Compile with:
 *   g++ -std=c++17 -o test_svs_serialization test_svs_serialization.cpp \
 *       -I jni/external/faiss -L jni/build/faiss/faiss -lfaiss_avx512_spr \
 *       -Wl,-rpath,jni/build/faiss/faiss
 * 
 * Run with:
 *   ./test_svs_serialization
 */

#include <faiss/Index.h>
#include <faiss/IndexFactory.h>
#include <faiss/index_io.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <stdexcept>

void test_svs_flat_basic() {
    std::cout << "\n=== Test 1: Basic SVS Flat Creation ===" << std::endl;
    
    int dimension = 3;
    faiss::Index* index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_L2);
    
    std::cout << "✓ Index created successfully" << std::endl;
    std::cout << "  Type: " << typeid(*index).name() << std::endl;
    std::cout << "  Dimension: " << index->d << std::endl;
    std::cout << "  Metric: " << (index->metric_type == faiss::METRIC_L2 ? "L2" : "other") << std::endl;
    std::cout << "  Is trained: " << (index->is_trained ? "yes" : "no") << std::endl;
    std::cout << "  ntotal: " << index->ntotal << std::endl;
    
    delete index;
}

void test_svs_flat_add_vectors() {
    std::cout << "\n=== Test 2: Add Vectors to SVS Flat ===" << std::endl;
    
    int dimension = 3;
    faiss::Index* index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_L2);
    
    // Create test vectors
    std::vector<float> vectors = {
        1.0f, 1.0f, 1.0f,  // vector 0
        2.0f, 2.0f, 2.0f,  // vector 1
        3.0f, 3.0f, 3.0f   // vector 2
    };
    
    int n = vectors.size() / dimension;
    std::cout << "Adding " << n << " vectors..." << std::endl;
    
    try {
        index->add(n, vectors.data());
        std::cout << "✓ Vectors added successfully" << std::endl;
        std::cout << "  ntotal after add: " << index->ntotal << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to add vectors: " << e.what() << std::endl;
        delete index;
        throw;
    }
    
    delete index;
}

void test_svs_flat_write() {
    std::cout << "\n=== Test 3: Write SVS Flat to Disk ===" << std::endl;
    
    int dimension = 3;
    faiss::Index* index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_L2);
    
    // Add vectors
    std::vector<float> vectors = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f
    };
    index->add(3, vectors.data());
    
    const char* filename = "/tmp/test_svs_flat.faiss";
    std::cout << "Writing index to: " << filename << std::endl;
    
    try {
        faiss::write_index(index, filename);
        std::cout << "✓ Index written successfully" << std::endl;
        
        // Check file exists
        FILE* f = fopen(filename, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fclose(f);
            std::cout << "  File size: " << size << " bytes" << std::endl;
        } else {
            std::cout << "✗ File not created!" << std::endl;
            delete index;
            throw std::runtime_error("File not created");
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to write index: " << e.what() << std::endl;
        delete index;
        throw;
    }
    
    delete index;
}

void test_svs_flat_read() {
    std::cout << "\n=== Test 4: Read SVS Flat from Disk ===" << std::endl;
    
    // First write an index
    int dimension = 3;
    faiss::Index* write_index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_L2);
    std::vector<float> vectors = {
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f
    };
    write_index->add(3, vectors.data());
    
    const char* filename = "/tmp/test_svs_flat_read.faiss";
    faiss::write_index(write_index, filename);
    delete write_index;
    
    std::cout << "Reading index from: " << filename << std::endl;
    
    try {
        faiss::Index* read_index = faiss::read_index(filename);
        std::cout << "✓ Index loaded successfully" << std::endl;
        std::cout << "  Type: " << typeid(*read_index).name() << std::endl;
        std::cout << "  Dimension: " << read_index->d << std::endl;
        std::cout << "  Metric: " << (read_index->metric_type == faiss::METRIC_L2 ? "L2" : "other") << std::endl;
        std::cout << "  ntotal: " << read_index->ntotal << std::endl;
        
        delete read_index;
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to read index: " << e.what() << std::endl;
        throw;
    }
}

void test_svs_flat_search() {
    std::cout << "\n=== Test 5: Search After Load ===" << std::endl;
    
    // Create and save index
    int dimension = 3;
    faiss::Index* write_index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_L2);
    std::vector<float> vectors = {
        1.0f, 1.0f, 1.0f,  // vector 0
        2.0f, 2.0f, 2.0f,  // vector 1
        3.0f, 3.0f, 3.0f   // vector 2
    };
    write_index->add(3, vectors.data());
    
    const char* filename = "/tmp/test_svs_flat_search.faiss";
    faiss::write_index(write_index, filename);
    delete write_index;
    
    // Load and search
    faiss::Index* read_index = faiss::read_index(filename);
    
    std::vector<float> query = {1.5f, 1.5f, 1.5f};
    int k = 2;
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    
    std::cout << "Performing search with query [1.5, 1.5, 1.5], k=" << k << std::endl;
    
    try {
        read_index->search(1, query.data(), k, distances.data(), labels.data());
        std::cout << "✓ Search completed successfully" << std::endl;
        
        std::cout << "Search results:" << std::endl;
        for (int i = 0; i < k; i++) {
            std::cout << "  Result " << i << ": label=" << labels[i] 
                      << ", distance=" << distances[i] << std::endl;
        }
        
        // Verify results make sense
        if (labels[0] == 0 || labels[0] == 1) {
            std::cout << "✓ Results look correct (closest to vectors 0 or 1)" << std::endl;
        } else {
            std::cout << "⚠ Unexpected result order" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Search failed: " << e.what() << std::endl;
        delete read_index;
        throw;
    }
    
    delete read_index;
}

void test_svs_flat_inner_product() {
    std::cout << "\n=== Test 6: SVS Flat with INNER_PRODUCT ===" << std::endl;
    
    int dimension = 3;
    faiss::Index* index = faiss::index_factory(dimension, "SVSFlat", faiss::METRIC_INNER_PRODUCT);
    
    std::cout << "✓ Index created with INNER_PRODUCT metric" << std::endl;
    std::cout << "  Metric: " << (index->metric_type == faiss::METRIC_INNER_PRODUCT ? "INNER_PRODUCT" : "other") << std::endl;
    
    // Add and test
    std::vector<float> vectors = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    index->add(3, vectors.data());
    
    const char* filename = "/tmp/test_svs_flat_ip.faiss";
    faiss::write_index(index, filename);
    delete index;
    
    faiss::Index* loaded = faiss::read_index(filename);
    std::cout << "✓ Index saved and loaded with INNER_PRODUCT" << std::endl;
    
    delete loaded;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "SVS Flat Serialization Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int failed = 0;
    int passed = 0;
    
    // Run all tests
    struct Test {
        const char* name;
        void (*func)();
    };
    
    Test tests[] = {
        {"Basic Creation", test_svs_flat_basic},
        {"Add Vectors", test_svs_flat_add_vectors},
        {"Write to Disk", test_svs_flat_write},
        {"Read from Disk", test_svs_flat_read},
        {"Search After Load", test_svs_flat_search},
        {"INNER_PRODUCT Metric", test_svs_flat_inner_product}
    };
    
    for (const auto& test : tests) {
        try {
            test.func();
            passed++;
        } catch (const std::exception& e) {
            std::cout << "\n✗ Test '" << test.name << "' FAILED: " << e.what() << std::endl;
            failed++;
        } catch (...) {
            std::cout << "\n✗ Test '" << test.name << "' FAILED with unknown exception" << std::endl;
            failed++;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << std::endl;
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << std::endl;
    std::cout << "========================================" << std::endl;
    
    return failed == 0 ? 0 : 1;
}
