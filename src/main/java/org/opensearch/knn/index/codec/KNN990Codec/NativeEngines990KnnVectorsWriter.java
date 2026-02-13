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

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.codec.nativeindex.ShardModelCache;
import org.opensearch.knn.index.codec.transfer.OffHeapFloatVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.FieldInfoExtractor.isDeferredLeanVecEnabled;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_INITIAL_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_LEANVEC_DIMS;
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QTY;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * A KNNVectorsWriter class for writing the vector data structures and flat vectors for Native Engines.
 */
@Log4j2
public class NativeEngines990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngines990KnnVectorsWriter.class);

    /**
     * Maximum number of vectors to use for training. If more are available, we sample uniformly.
     * 1M vectors at 768D float32 ≈ 2.9 GB off-heap — keeps memory bounded (C4 fix).
     */
    private static final int MAX_TRAINING_VECTORS = 1_000_000;

    /**
     * Minimum allowed training threshold to prevent accidental training on tiny segments.
     */
    private static final int MIN_TRAINING_THRESHOLD = 1000;

    /**
     * Sentinel return values for training methods to distinguish non-failure conditions
     * from real training failures. These must NOT trigger circuit breaker (K-R1-5 fix).
     */
    private static final byte[] TRAINING_INTERRUPTED = new byte[0];

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private KNN990QuantizationStateWriter quantizationStateWriter;
    private LeanVecModelWriter leanVecModelWriter;
    private final List<NativeEngineFieldVectorsWriter<?>> fields = new ArrayList<>();
    private boolean finished;
    private final Integer approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Add new field for indexing.
     * @param fieldInfo {@link FieldInfo}
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        final NativeEngineFieldVectorsWriter<?> newField = NativeEngineFieldVectorsWriter.create(
            fieldInfo,
            flatVectorsWriter.addField(fieldInfo),
            segmentWriteState.infoStream
        );
        fields.add(newField);
        return newField;
    }

    /**
     * Flush all buffered data on disk. This is not fsync. This is lucene flush.
     *
     * @param maxDoc int
     * @param sortMap {@link Sorter.DocMap}
     */
    @Override
    public void flush(int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            final FieldInfo fieldInfo = field.getFieldInfo();
            final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
            int totalLiveDocs = field.getVectors().size();
            if (totalLiveDocs == 0) {
                log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
                continue;
            }
            final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getVectorValuesSupplier(
                vectorDataType,
                field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                field.getVectors()
            );
            // K-16: Count vectors for cumulative threshold BEFORE shouldSkipBuildingVectorDataStructure.
            // Small flushes must still be counted even when graph building is skipped.
            if (isDeferredLeanVecEnabled(fieldInfo)) {
                String shardId = getShardId();
                ShardModelCache cache = ShardModelCache.getInstance(shardId);
                long cumulative = cache.addVectors(fieldInfo.name, totalLiveDocs);
                log.info("[Flush] Counted {} vectors for field '{}', cumulative={}", totalLiveDocs, fieldInfo.name, cumulative);
            }

            final QuantizationState quantizationState = train(field.getFieldInfo(), knnVectorValuesSupplier, totalLiveDocs);
            // should skip graph building only for non quantization use case and if threshold is met
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                    fieldInfo.name,
                    totalLiveDocs,
                    approximateThreshold
                );
                continue;
            }

            // Load model from shard cache for flush (post-training segments use LeanVec)
            byte[] shardModelBlob = null;
            if (isDeferredLeanVecEnabled(fieldInfo)) {
                String shardId = getShardId();
                ShardModelCache cache = ShardModelCache.getInstance(shardId);
                // Single atomic read of blob+quality to avoid race between getModel/getModelQuality (W-10 fix)
                ShardModelCache.CachedModel cached = cache.getCachedModel(fieldInfo.name);
                if (cached != null) {
                    shardModelBlob = cached.blobCopy();
                    log.info("[Flush] Encoding=LeanVec ({} model) for field '{}' ({} vectors, segment={})",
                        cached.quality(), fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
                } else {
                    log.info("[Flush] Encoding=LVQ (no trained model) for field '{}' ({} vectors, segment={})",
                        fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
                }
            }

            final NativeIndexWriter writer = NativeIndexWriter.getWriter(
                fieldInfo,
                segmentWriteState,
                quantizationState,
                nativeIndexBuildStrategyFactory,
                shardModelBlob
            );

            StopWatch stopWatch = new StopWatch().start();
            writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);

            // Persist model blob to segment file during flush so it survives node restart (C-3 fix)
            if (shardModelBlob != null) {
                writeLeanVecModelToSegment(fieldInfo.getFieldNumber(), shardModelBlob);
            }

            long time_in_millis = stopWatch.stop().totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
            log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType,
            fieldInfo,
            mergeState
        );
        int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        final QuantizationState quantizationState = train(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
        // should skip graph building only for non quantization use case and if threshold is met
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name,
                totalLiveDocs,
                approximateThreshold
            );
            return;
        }

        // Deferred LeanVec training: check if we should train or reuse existing model
        byte[] shardModelBlob = null;
        if (isDeferredLeanVecEnabled(fieldInfo)) {
            shardModelBlob = maybeTriggerLeanVecTraining(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
        }

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo,
            segmentWriteState,
            quantizationState,
            nativeIndexBuildStrategyFactory,
            shardModelBlob
        );

        StopWatch stopWatch = new StopWatch().start();

        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);

        // If we trained or propagated a model, write it to the output segment file
        if (shardModelBlob != null) {
            writeLeanVecModelToSegment(fieldInfo.getFieldNumber(), shardModelBlob);
        }

        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    /**
     * Called once at the end before close
     */
    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("NativeEnginesKNNVectorsWriter is already finished");
        }
        finished = true;
        if (quantizationStateWriter != null) {
            quantizationStateWriter.writeFooter();
        }
        if (leanVecModelWriter != null) {
            leanVecModelWriter.writeFooter();
        }
        flatVectorsWriter.finish();
    }

    @Override
    public void close() throws IOException {
        // C-R3-3/W-R3-1: Use IOUtils.close() to ensure all resources are released even if
        // earlier closes throw. LeanVecModelWriter implements Closeable. Wrap quantization
        // state writer's closeOutput in a Closeable lambda since it doesn't implement Closeable.
        java.io.Closeable quantizationCloseable = quantizationStateWriter != null
            ? () -> quantizationStateWriter.closeOutput()
            : null;
        IOUtils.close(quantizationCloseable, leanVecModelWriter, flatVectorsWriter);
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + fields.stream()
            .mapToLong(NativeEngineFieldVectorsWriter::ramBytesUsed)
            .sum();
    }

    private QuantizationState train(
        final FieldInfo fieldInfo,
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        final int totalLiveDocs
    ) throws IOException {

        final QuantizationService quantizationService = QuantizationService.getInstance();
        final QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);
        QuantizationState quantizationState = null;
        if (quantizationParams != null && totalLiveDocs > 0) {
            initQuantizationStateWriterIfNecessary();
            quantizationState = quantizationService.train(quantizationParams, knnVectorValuesSupplier, totalLiveDocs);
            quantizationStateWriter.writeState(fieldInfo.getFieldNumber(), quantizationState);
        }

        return quantizationState;
    }

    /**
     * The {@link KNNVectorValues} will be exhausted after this function run. So make sure that you are not sending the
     * vectorsValues object which you plan to use later.
     *
     * Note: The knnVectorValuesSupplier passed to training and index building must produce independent iterators
     * that can be consumed separately (each call to .get() returns a fresh iterator).
     */
    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }

    private void initQuantizationStateWriterIfNecessary() throws IOException {
        if (quantizationStateWriter == null) {
            quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
            quantizationStateWriter.writeHeader(segmentWriteState);
        }
    }

    private void initLeanVecModelWriterIfNecessary() throws IOException {
        if (leanVecModelWriter == null) {
            leanVecModelWriter = new LeanVecModelWriter(segmentWriteState);
            leanVecModelWriter.writeHeader(segmentWriteState);
        }
    }

    private boolean shouldSkipBuildingVectorDataStructure(final long docCount) {
        if (approximateThreshold < 0) {
            return true;
        }
        return docCount < approximateThreshold;
    }

    // ---- Deferred LeanVec training helpers ----

    /**
     * Gets the final training threshold for deferred LeanVec training (W6: NumberFormatException safety, S9: min threshold).
     */
    private static int getFinalTrainingThreshold(FieldInfo fieldInfo) {
        String val = fieldInfo.attributes().get(DEFERRED_TRAINING_THRESHOLD);
        if (val == null) {
            return DEFERRED_TRAINING_DEFAULT_THRESHOLD;
        }
        try {
            int threshold = Integer.parseInt(val);
            if (threshold < MIN_TRAINING_THRESHOLD) {
                log.warn("Final training threshold {} below minimum {}, using minimum", threshold, MIN_TRAINING_THRESHOLD);
                return MIN_TRAINING_THRESHOLD;
            }
            return threshold;
        } catch (NumberFormatException e) {
            log.warn("Invalid final training threshold '{}', using default {}", val, DEFERRED_TRAINING_DEFAULT_THRESHOLD);
            return DEFERRED_TRAINING_DEFAULT_THRESHOLD;
        }
    }

    /**
     * Gets the initial training threshold for two-threshold deferred LeanVec training.
     * Falls back to DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD (10,000).
     */
    private static int getInitialTrainingThreshold(FieldInfo fieldInfo) {
        String val = fieldInfo.attributes().get(DEFERRED_TRAINING_INITIAL_THRESHOLD);
        if (val == null) {
            return DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;
        }
        try {
            int threshold = Integer.parseInt(val);
            if (threshold < MIN_TRAINING_THRESHOLD) {
                log.warn("Initial training threshold {} below minimum {}, using minimum", threshold, MIN_TRAINING_THRESHOLD);
                return MIN_TRAINING_THRESHOLD;
            }
            return threshold;
        } catch (NumberFormatException e) {
            log.warn("Invalid initial training threshold '{}', using default {}", val, DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD);
            return DEFERRED_TRAINING_DEFAULT_INITIAL_THRESHOLD;
        }
    }

    /**
     * Gets the target LeanVec dimensions (W6: NumberFormatException safety).
     */
    private static int getLeanVecDimensions(FieldInfo fieldInfo) {
        String val = fieldInfo.attributes().get(DEFERRED_TRAINING_LEANVEC_DIMS);
        if (val == null) {
            return 0;
        }
        try {
            return Integer.parseInt(val);
        } catch (NumberFormatException e) {
            log.warn("Invalid LeanVec dimensions '{}', returning 0", val);
            return 0;
        }
    }

    /**
     * Gets the vector dimension from field attributes (W6: NumberFormatException safety).
     */
    private static int getDimension(FieldInfo fieldInfo) {
        String val = fieldInfo.attributes().get(DIMENSION);
        if (val == null) {
            return 0;
        }
        try {
            return Integer.parseInt(val);
        } catch (NumberFormatException e) {
            log.warn("Invalid dimension '{}', returning 0", val);
            return 0;
        }
    }

    /**
     * Two-threshold training decision logic for deferred LeanVec training.
     *
     * State machine: NONE → INITIAL (10K vectors) → FINAL (100K vectors).
     * Returns the model blob to use for this merge, or null for LVQ fallback.
     *
     * Decision tree:
     * - FINAL model cached → use it
     * - INITIAL model cached + merge >= final_threshold → retrain as FINAL
     * - INITIAL model cached + merge < final_threshold → use INITIAL
     * - No model + cumulative < initial_threshold → LVQ (not enough vectors yet)
     * - No model + merge >= final_threshold → train FINAL directly (skip INITIAL)
     * - No model + merge >= initial_threshold → train INITIAL
     * - No model + merge < initial_threshold → LVQ (wait for bigger merge)
     */
    private byte[] maybeTriggerLeanVecTraining(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs
    ) throws IOException {
        String shardId = getShardId();
        ShardModelCache cache = ShardModelCache.getInstance(shardId);
        int initialThreshold = getInitialTrainingThreshold(fieldInfo);
        int finalThreshold = getFinalTrainingThreshold(fieldInfo);

        // 1. Read cached model — single atomic read (C-2 fix)
        ShardModelCache.CachedModel cached = cache.getCachedModel(fieldInfo.name);
        ShardModelCache.ModelQuality quality = (cached != null) ? cached.quality() : ShardModelCache.ModelQuality.NONE;

        // 2. FINAL model exists — always use it
        if (quality == ShardModelCache.ModelQuality.FINAL) {
            log.info("[Merge] Encoding=LeanVec (FINAL model) for field '{}' ({} vectors, segment={})",
                fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
            return cached.blobCopy();
        }

        // 3. INITIAL model exists — check if this merge can upgrade to FINAL
        if (quality == ShardModelCache.ModelQuality.INITIAL) {
            if (totalLiveDocs >= finalThreshold) {
                // This merge is large enough to train the FINAL model
                return tryTrainModel(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, cache,
                    ShardModelCache.ModelQuality.FINAL, finalThreshold);
            }
            // Merge too small to upgrade — use existing INITIAL model
            log.info("[Merge] Encoding=LeanVec (INITIAL model) for field '{}' ({} vectors, segment={})",
                fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
            return cached.blobCopy();
        }

        // 4. No model yet — seed counter on first merge after restart
        if (cache.tryMarkCounterSeeded(fieldInfo.name)) {
            seedCounterFromCommittedSegments(fieldInfo, cache);
        }

        // 5. Cumulative threshold check — need enough total vectors before training
        long cumulativeCount = cache.getCumulativeVectorCount(fieldInfo.name);
        if (cumulativeCount < initialThreshold) {
            log.info("[Merge] Encoding=LVQ (cumulative {} < initial threshold {}) for field '{}' ({} vectors, segment={})",
                cumulativeCount, initialThreshold, fieldInfo.name, totalLiveDocs,
                segmentWriteState.segmentInfo.name);
            return null;
        }

        // 6. Cumulative threshold met — determine which model to train based on merge size
        if (totalLiveDocs >= finalThreshold) {
            // Skip INITIAL, go straight to FINAL
            return tryTrainModel(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, cache,
                ShardModelCache.ModelQuality.FINAL, finalThreshold);
        }
        if (totalLiveDocs >= initialThreshold) {
            // Train INITIAL model
            return tryTrainModel(fieldInfo, knnVectorValuesSupplier, totalLiveDocs, cache,
                ShardModelCache.ModelQuality.INITIAL, initialThreshold);
        }

        // 7. Merge too small for either threshold
        log.info("[Merge] Encoding=LVQ (merge {} < initial threshold {}) for field '{}' (segment={})",
            totalLiveDocs, initialThreshold, fieldInfo.name, segmentWriteState.segmentInfo.name);
        return null;
    }

    /**
     * Attempts to train a LeanVec model at the specified quality level.
     * Handles circuit breaker, locking, training, caching, and fallback.
     *
     * @param fieldInfo              the field being indexed
     * @param knnVectorValuesSupplier supplier for merge vectors
     * @param totalLiveDocs          number of vectors in the merge
     * @param cache                  the shard model cache
     * @param targetQuality          the quality level to train (INITIAL or FINAL)
     * @param threshold              the threshold that was met
     * @return model blob if training succeeded, existing model if available, or null for LVQ fallback
     */
    private byte[] tryTrainModel(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs,
        ShardModelCache cache,
        ShardModelCache.ModelQuality targetQuality,
        int threshold
    ) throws IOException {
        // Circuit breaker check
        if (cache.isTrainingSuppressed(fieldInfo.name, targetQuality)) {
            log.warn("[Merge] {} training suppressed by circuit breaker for field '{}' — using fallback",
                targetQuality, fieldInfo.name);
            // Return existing model if available (INITIAL), else null (LVQ)
            return cache.getModel(fieldInfo.name);
        }
        if (KNNSettings.isCircuitBreakerTriggered()) {
            log.warn("[Merge] KNN circuit breaker triggered, deferring {} training for field '{}'",
                targetQuality, fieldInfo.name);
            return cache.getModel(fieldInfo.name);
        }

        // Acquire training lock (non-blocking, 100ms timeout)
        ReentrantLock lock = cache.getTrainingLock(fieldInfo.name);
        boolean acquired;
        try {
            acquired = lock.tryLock(100, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("[Merge] Interrupted while acquiring training lock for field '{}'", fieldInfo.name);
            return cache.getModel(fieldInfo.name);
        }
        if (!acquired) {
            // Another merge is training — use existing model if available, else LVQ
            byte[] existing = cache.getModel(fieldInfo.name);
            if (existing != null) {
                log.info("[Merge] Encoding=LeanVec (lock busy, using existing model) for field '{}' (segment={})",
                    fieldInfo.name, segmentWriteState.segmentInfo.name);
                return existing;
            }
            log.info("[Merge] Encoding=LVQ (training lock busy) for field '{}' (segment={})",
                fieldInfo.name, segmentWriteState.segmentInfo.name);
            return null;
        }

        try {
            // Double-check: model may have been upgraded while waiting for lock
            // Single atomic read — no torn state possible (C-2 fix)
            ShardModelCache.CachedModel current = cache.getCachedModel(fieldInfo.name);
            ShardModelCache.ModelQuality currentQuality =
                (current != null) ? current.quality() : ShardModelCache.ModelQuality.NONE;
            if (!currentQuality.isUpgradeableTo(targetQuality)) {
                // Already at or above target quality
                if (current != null) {
                    log.info("[Merge] Encoding=LeanVec ({} model, upgraded while waiting) for field '{}' (segment={})",
                        currentQuality, fieldInfo.name, segmentWriteState.segmentInfo.name);
                    return current.blobCopy();
                }
            }

            // Train — increment both aggregate and per-quality counters
            KNNCounter.DEFERRED_TRAINING_REQUESTS.increment();
            incrementQualityCounter(targetQuality, true, false);
            StopWatch stopWatch = new StopWatch().start();
            log.info("[Merge] TRAINING STARTED ({}): field='{}', mergeVectors={}, threshold={}, segment={}",
                targetQuality, fieldInfo.name, totalLiveDocs, threshold,
                segmentWriteState.segmentInfo.name);

            byte[] modelBlob = trainLeanVecModel(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);

            long trainingMs = stopWatch.stop().totalTime().millis();

            if (modelBlob != null && modelBlob.length > 0) {
                boolean stored = cache.putModel(fieldInfo.name, modelBlob, targetQuality);
                if (!stored) {
                    // Monotonicity guard rejected the store — another thread already upgraded
                    log.info("[Merge] Model store rejected (higher quality exists) for field '{}' — using cached",
                        fieldInfo.name);
                    return cache.getModel(fieldInfo.name);
                }
                KNNCounter.DEFERRED_TRAINING_SUCCESS.increment();
                incrementQualityCounter(targetQuality, false, true);
                log.info("[Merge] TRAINING COMPLETE ({}, {}ms): Encoding=LeanVec for field '{}' (segment={})",
                    targetQuality, trainingMs, fieldInfo.name, segmentWriteState.segmentInfo.name);
                if (trainingMs > 2000) {
                    log.warn("[Merge] Training took {}ms (>2s) for field '{}' — may impact merge throughput",
                        trainingMs, fieldInfo.name);
                }
                return modelBlob;
            } else if (modelBlob == TRAINING_INTERRUPTED) {
                KNNCounter.DEFERRED_TRAINING_INTERRUPTED.increment();
                log.info("[Merge] Training interrupted for field '{}', using fallback ({}ms)",
                    fieldInfo.name, trainingMs);
                // Return existing model if we have one (INITIAL), else null (LVQ)
                return cache.getModel(fieldInfo.name);
            } else {
                cache.recordFailure(fieldInfo.name, targetQuality);
                KNNCounter.DEFERRED_TRAINING_ERRORS.increment();
                incrementQualityCounter(targetQuality, false, false);
                log.warn("[Merge] TRAINING FAILED ({}, {}ms): fallback for field '{}' (segment={})",
                    targetQuality, trainingMs, fieldInfo.name, segmentWriteState.segmentInfo.name);
                // Return existing model if we have one (INITIAL), else null (LVQ)
                return cache.getModel(fieldInfo.name);
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * Increments per-quality training counters for stats endpoint visibility.
     *
     * @param quality  the target quality level
     * @param isRequest true to increment the request counter (called at training start)
     * @param isSuccess true to increment success counter, false to increment error counter
     *                  (only used when isRequest is false)
     */
    private static void incrementQualityCounter(ShardModelCache.ModelQuality quality, boolean isRequest, boolean isSuccess) {
        if (quality == ShardModelCache.ModelQuality.INITIAL) {
            if (isRequest) {
                KNNCounter.DEFERRED_TRAINING_INITIAL_REQUESTS.increment();
            } else if (isSuccess) {
                KNNCounter.DEFERRED_TRAINING_INITIAL_SUCCESS.increment();
            } else {
                KNNCounter.DEFERRED_TRAINING_INITIAL_ERRORS.increment();
            }
        } else if (quality == ShardModelCache.ModelQuality.FINAL) {
            if (isRequest) {
                KNNCounter.DEFERRED_TRAINING_FINAL_REQUESTS.increment();
            } else if (isSuccess) {
                KNNCounter.DEFERRED_TRAINING_FINAL_SUCCESS.increment();
            } else {
                KNNCounter.DEFERRED_TRAINING_FINAL_ERRORS.increment();
            }
        }
    }

    // NOTE: After node restart, the ShardModelCache is warmed by KNNPlugin.afterIndexShardStarted()
    // which loads models from .knnlvm segment files and seeds the cumulative counter.
    // This eliminates the LVQ fallback window after restart (O-14 fix).

    /**
     * Writes a LeanVec model blob to the output segment file.
     *
     * C-R3-1 note: The .knnlvm file is automatically tracked by Lucene's TrackingDirectoryWrapper
     * because it is created via segmentWriteState.directory.createOutput() in LeanVecModelWriter.
     * After the codec finishes, IndexWriter calls si.setFiles(wrapper.getCreatedFiles()) which
     * includes our file. This is the same pattern used by engine files (.faiss) and quantization
     * state files (.knnq) — none of them call segmentInfo.addFile() explicitly.
     */
    private void writeLeanVecModelToSegment(int fieldNumber, byte[] modelBlob) throws IOException {
        initLeanVecModelWriterIfNecessary();
        leanVecModelWriter.writeModel(fieldNumber, modelBlob);
    }

    /**
     * Trains a LeanVec model using the vectors from the merge.
     *
     * Steps:
     * 1. Read vectors from KNNVectorValues into off-heap memory (capped at MAX_TRAINING_VECTORS)
     * 2. Build training parameters (index_description with LeanVec encoding)
     * 3. Call JNIService.trainIndex() to produce a model blob
     * 4. Return the model blob for immediate use
     *
     * Note: JNICommons.storeVectorData() allocates a std::vector<float>* and returns its address as jlong.
     * JNIService.trainIndex() casts this back to std::vector<float>* — the types match (C1 clarification).
     */
    private byte[] trainLeanVecModel(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs
    ) throws IOException {
        int dimension = getDimension(fieldInfo);
        int leanvecDims = getLeanVecDimensions(fieldInfo);
        if (dimension <= 0 || leanvecDims <= 0) {
            // W-R3-7: Record failure on early return so circuit breaker and stats are accurate
            log.warn("Cannot train LeanVec: invalid dimensions (dim={}, leanvecDims={})", dimension, leanvecDims);
            return null;
        }

        // W-R3-7: Track timing with try-finally to ensure StopWatch is always stopped
        StopWatch stopWatch = new StopWatch().start();
        boolean stopped = false;
        try {
            // Build training parameters
            String indexDescription = getIndexDescriptionFromFieldInfo(fieldInfo);
            Map<String, Object> trainParameters = new HashMap<>();
            trainParameters.put(INDEX_DESCRIPTION_PARAMETER, indexDescription);
            trainParameters.put(SPACE_TYPE, fieldInfo.attributes().getOrDefault(SPACE_TYPE, "l2"));
            trainParameters.put(INDEX_THREAD_QTY, KNNSettings.getIndexThreadQty());
            trainParameters.put(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());

            // Cap training vectors at MAX_TRAINING_VECTORS to bound off-heap memory (C4 fix)
            int trainingCount = Math.min(totalLiveDocs, MAX_TRAINING_VECTORS);
            // Calculate sampling step for uniform sampling when totalLiveDocs > MAX_TRAINING_VECTORS (W5)
            int sampleStep = totalLiveDocs > MAX_TRAINING_VECTORS ? totalLiveDocs / MAX_TRAINING_VECTORS : 1;

            int bytesPerVector = dimension * Float.BYTES;
            try (OffHeapFloatVectorTransfer vectorTransfer = new OffHeapFloatVectorTransfer(bytesPerVector, trainingCount)) {
                KNNVectorValues<?> knnVectorValues = knnVectorValuesSupplier.get();
                knnVectorValues.nextDoc();

                int transferred = 0;
                int docIndex = 0;
                while (knnVectorValues.docId() != DocIdSetIterator.NO_MORE_DOCS && transferred < trainingCount) {
                    // W8: Check for thread interruption during long training transfers
                    if (Thread.currentThread().isInterrupted()) {
                        log.warn("LeanVec training interrupted for field {}", fieldInfo.name);
                        return TRAINING_INTERRUPTED;
                    }

                    if (docIndex % sampleStep == 0) {
                        // K-R2-8 FIX: Clone to prevent buffer reuse corruption in batch transfer
                        float[] vector = ((float[]) knnVectorValues.getVector()).clone();
                        vectorTransfer.transfer(vector, true);
                        transferred++;
                    }
                    docIndex++;
                    knnVectorValues.nextDoc();
                }
                vectorTransfer.flush(true);

                long vectorAddress = vectorTransfer.getVectorAddress();

                // TODO: Consider replacing AccessController.doPrivileged when codebase moves to module system (W1)
                byte[] modelBlob = AccessController.doPrivileged((PrivilegedAction<byte[]>) () -> {
                    return JNIService.trainIndex(trainParameters, dimension, vectorAddress, KNNEngine.FAISS);
                });

                if (modelBlob == null || modelBlob.length == 0) {
                    log.error("LeanVec training produced empty model blob for field {}", fieldInfo.name);
                    return null;
                }

                long timeMs = stopWatch.stop().totalTime().millis();
                stopped = true;
                log.info(
                    "[Merge] LeanVec training complete for field {} ({} vectors sampled from {}, {}ms, blob={}KB)",
                    fieldInfo.name,
                    transferred,
                    totalLiveDocs,
                    timeMs,
                    modelBlob.length / 1024
                );

                return modelBlob;
            }
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            // W-13 fix: Only catch expected training failures; let OOM/Error propagate
            log.error("Failed to train LeanVec model for field {}: {}", fieldInfo.name, e.getMessage(), e);
            return null;
        } finally {
            // W-R3-7: Ensure StopWatch is stopped on all exit paths (early returns, exceptions)
            if (!stopped) {
                try { stopWatch.stop(); } catch (IllegalStateException ignored) { }
            }
        }
    }

    /**
     * Seeds the cumulative vector counter from committed segment metadata.
     * Called on the first merge after restart to avoid a long LVQ fallback window.
     * Uses SegmentInfos.readLatestCommit() which reads a single small file (no locking).
     *
     * Note: maxDoc() - delCount() counts all live documents, not just vector-bearing ones.
     * For sparse vector fields this overcounts (triggers training earlier than necessary, which is harmless).
     */
    private void seedCounterFromCommittedSegments(FieldInfo fieldInfo, ShardModelCache cache) {
        try {
            Directory rawDir = FilterDirectory.unwrap(segmentWriteState.directory);
            SegmentInfos infos = SegmentInfos.readLatestCommit(rawDir);
            long totalVectors = 0;
            for (SegmentCommitInfo sci : infos) {
                totalVectors += sci.info.maxDoc() - sci.getDelCount();
            }
            cache.seedVectorCount(fieldInfo.name, totalVectors);
            log.info("[Merge] Seeded cumulative counter from {} committed segments: {} vectors for field '{}'",
                infos.size(), totalVectors, fieldInfo.name);
        } catch (IOException e) {
            log.warn("[Merge] Failed to seed counter from committed segments for field '{}': {}", fieldInfo.name, e.getMessage());
        }
    }

    /**
     * Extracts the index_description from field attributes PARAMETERS JSON.
     * Throws IllegalStateException if parameters cannot be parsed or index_description is missing (W2 fix).
     */
    private static String getIndexDescriptionFromFieldInfo(FieldInfo fieldInfo) {
        String parametersString = fieldInfo.attributes().get(PARAMETERS);
        if (parametersString != null) {
            try {
                @SuppressWarnings("unchecked")
                Map<String, Object> params = org.opensearch.common.xcontent.XContentHelper.createParser(
                    org.opensearch.core.xcontent.NamedXContentRegistry.EMPTY,
                    org.opensearch.core.xcontent.DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new org.opensearch.core.common.bytes.BytesArray(parametersString),
                    org.opensearch.core.xcontent.MediaTypeRegistry.JSON  // W-R3-8: explicit JSON, not runtime default
                ).map();
                Object desc = params.get(INDEX_DESCRIPTION_PARAMETER);
                if (desc != null) {
                    return desc.toString();
                }
            } catch (Exception e) {
                throw new IllegalStateException(
                    "Failed to parse parameters for index_description on field " + fieldInfo.name + ": " + e.getMessage(), e
                );
            }
        }
        throw new IllegalStateException(
            "Cannot determine index_description for deferred LeanVec training on field " + fieldInfo.name
            + ": no parameters attribute found"
        );
    }

    /**
     * Derives a stable shard identifier from the segment write state directory.
     * Unwraps FilterDirectory wrappers to get the underlying FSDirectory path,
     * which produces a consistent key like "/data/nodes/0/indices/&lt;uuid&gt;/&lt;shard&gt;/index".
     *
     * This must match the key used by ShardModelCache.removeInstancesForShard() in KNNPlugin
     * to prevent memory leaks (C-1 fix). L-R1-4: Use FilterDirectory.unwrap() for consistency.
     */
    private String getShardId() {
        Directory dir = FilterDirectory.unwrap(segmentWriteState.directory);
        if (dir instanceof org.apache.lucene.store.FSDirectory) {
            return ((org.apache.lucene.store.FSDirectory) dir).getDirectory().toString();
        }
        return segmentWriteState.directory.toString();
    }
}
