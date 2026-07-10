/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.jni.KNNLibraryLoader;

import java.util.Map;

/**
 * JNI binding for the isolated {@code libopensearchknn_faiss_upstream} native library, which statically
 * embeds its own upstream faiss (vendored at its own pinned commit — newer than the k-NN pin) with all
 * symbols private. Reached only through {@link FaissUpstreamEngineProvider}; main code never references
 * this class, and the library loads lazily on first native use.
 *
 * <p>The native surface mirrors the engine's build/search lifecycle: init (an accumulation context —
 * IVF training happens at write time over the accumulated vectors), insert batches, write (train + add +
 * serialize), load, top-k query, free.
 */
public class FaissUpstreamService {

    /** Base name of the JNI library; variant suffixes (_avx512_spr, etc.) are appended by the loader. */
    private static final String FAISS_UPSTREAM_JNI_LIBRARY_NAME = "opensearchknn_faiss_upstream";

    static {
        // System.loadLibrary is centralized in KNNLibraryLoader (enforced by the validateLibraryUsage task).
        KNNLibraryLoader.loadLibraryByVariant(FAISS_UPSTREAM_JNI_LIBRARY_NAME);
        initLibrary();
        // Mark the registered engine initialized, mirroring how FaissService marks FAISS initialized.
        KNNEngine.getEngine(FaissUpstreamConstants.FAISS_UPSTREAM_ENGINE_NAME).setInitialized(true);
    }

    public static native long initIndex(long numDocs, int dim, Map<String, Object> parameters);

    public static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);

    public static native void writeIndex(long indexAddress, IndexOutputWithBuffer output);

    public static native long loadIndexWithStream(IndexInputWithBuffer readStream);

    public static native KNNQueryResult[] queryIndex(long indexPointer, float[] queryVector, int k, Map<String, ?> methodParameters);

    public static native void free(long indexPointer);

    public static native void initLibrary();
}
