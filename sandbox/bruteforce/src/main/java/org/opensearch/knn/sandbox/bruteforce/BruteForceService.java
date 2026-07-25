/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.KNNLibraryLoader;

import java.util.Map;

/**
 * JNI binding for {@code libopensearchknn_bruteforce}. The library loads on first use, through
 * {@link KNNLibraryLoader}, never at plugin startup.
 */
final class BruteForceService {

    private static final String BRUTEFORCE_JNI_LIBRARY_NAME = "opensearchknn_bruteforce";

    static {
        KNNLibraryLoader.loadLibraryByVariant(BRUTEFORCE_JNI_LIBRARY_NAME);
        KNNEngine.getEngine(BruteForceLibrary.ENGINE_NAME).setInitialized(true);
    }

    private BruteForceService() {}

    static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress);

    static native void writeIndex(long indexAddress, IndexOutputWithBuffer output);

    static native long loadIndexWithStream(IndexInputWithBuffer readStream);

    static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k, Map<String, ?> methodParameters);

    static native void free(long indexPointer);
}
