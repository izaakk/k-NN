/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#ifndef OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H
#define OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H

#include "faiss/impl/io.h"
#include "jni_util.h"
#include "native_engines_stream_support.h"
#include "parameter_utils.h"

#include <jni.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace knn_jni {
namespace stream {



/**
 * A glue component inheriting IOReader to be passed down to Faiss library.
 * This will then indirectly call the mediator component and eventually read required bytes from Lucene's IndexInput.
 */
class FaissOpenSearchIOReader final : public faiss::IOReader {
 public:
  explicit FaissOpenSearchIOReader(NativeEngineIndexInputMediator *_mediator)
      : faiss::IOReader(),
        mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
    name = "FaissOpenSearchIOReader";
  }

  size_t operator()(void *ptr, size_t size, size_t nitems) final {
    const auto total_bytes = size * nitems;
    if (total_bytes == 0 || size == 0) {
      return 0;
    }

    // Don't pre-check remainingBytes() - it may not be accurate due to buffering
    // or timing issues. Instead, try to read and handle exceptions.
    // The mediator's copyBytes() will loop until all bytes are read or throw IOException.
    try {
      mediator->copyBytes(total_bytes, (uint8_t *) ptr);
      // Successfully read all requested bytes
      return nitems;
    } catch (const std::exception& e) {
      // Read failed - could be EOF or other error
      // Check if any data is available for partial read
      int64_t available = mediator->remainingBytes();
      if (available > 0) {
        // Try to read what's available
        size_t items_available = available / size;
        if (items_available > 0) {
          size_t bytes_available = items_available * size;
          try {
            mediator->copyBytes(bytes_available, (uint8_t *) ptr);
            return items_available;
          } catch (...) {
            // Even partial read failed
            return 0;
          }
        }
      }
      // No data available or read failed
      return 0;
    }
  }

  int filedescriptor() final {
    throw std::runtime_error("filedescriptor() is not supported in FaissOpenSearchIOReader.");
  }

 private:
  NativeEngineIndexInputMediator *mediator;
};  // class FaissOpenSearchIOReader


/**
 * A glue component inheriting IOWriter to delegate IO processing down to the given
 * mediator. The mediator is expected to do write bytes via the provided Lucene's IndexOutput.
 */
class FaissOpenSearchIOWriter final : public faiss::IOWriter {
 public:
  explicit FaissOpenSearchIOWriter(NativeEngineIndexOutputMediator *_mediator)
      : faiss::IOWriter(),
        mediator(knn_jni::util::ParameterCheck::require_non_null(_mediator, "mediator")) {
    name = "FaissOpenSearchIOWriter";
  }

  size_t operator()(const void *ptr, size_t size, size_t nitems) final {
    const auto writeBytes = size * nitems;
    if (writeBytes > 0) {
      mediator->writeBytes(reinterpret_cast<const uint8_t *>(ptr), writeBytes);
    }
    return nitems;
  }

  // return a file number that can be memory-mapped
  int filedescriptor() final {
    throw std::runtime_error("filedescriptor() is not supported in FaissOpenSearchIOWriter.");
  }

  void flush() {
    mediator->flush();
  }

 private:
  NativeEngineIndexOutputMediator *mediator;
};  // class FaissOpenSearchIOWriter



}
}

#endif //OPENSEARCH_KNN_JNI_FAISS_STREAM_SUPPORT_H
