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
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
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
import static org.opensearch.knn.common.KNNConstants.DEFERRED_TRAINING_DEFAULT_THRESHOLD;
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
    private static final byte[] TRAINING_INSUFFICIENT_VECTORS = new byte[0];

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
                shardModelBlob = cache.getModel(fieldInfo.name);
                if (shardModelBlob != null) {
                    log.info("[Flush] Encoding=LeanVec (trained model available) for field '{}' ({} vectors, segment={})",
                        fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
                } else {
                    log.info("[Flush] Encoding=LVQ (pre-training fallback) for field '{}' ({} vectors, segment={})",
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
     * Gets the training threshold for deferred LeanVec training (W6: NumberFormatException safety, S9: min threshold).
     */
    private static int getTrainingThreshold(FieldInfo fieldInfo) {
        String val = fieldInfo.attributes().get(DEFERRED_TRAINING_THRESHOLD);
        if (val == null) {
            return DEFERRED_TRAINING_DEFAULT_THRESHOLD;
        }
        try {
            int threshold = Integer.parseInt(val);
            if (threshold < MIN_TRAINING_THRESHOLD) {
                log.warn("Training threshold {} below minimum {}, using minimum", threshold, MIN_TRAINING_THRESHOLD);
                return MIN_TRAINING_THRESHOLD;
            }
            return threshold;
        } catch (NumberFormatException e) {
            log.warn("Invalid training threshold '{}', using default {}", val, DEFERRED_TRAINING_DEFAULT_THRESHOLD);
            return DEFERRED_TRAINING_DEFAULT_THRESHOLD;
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
     * Checks if training should trigger and does so if appropriate.
     * Returns the shard model blob (from cache, input segments, or freshly trained), or null if no model available.
     *
     * Uses cumulative vector counting across flushes to determine when training should fire.
     * Training reads vectors from ALL committed segments via DirectoryReader to meet the SVS team's
     * requirement that training uses at least `threshold` vectors.
     *
     * Uses ShardModelCache for per-field locks (C3 fix) and circuit breaker (C6 fix).
     */
    private byte[] maybeTriggerLeanVecTraining(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        int totalLiveDocs
    ) throws IOException {
        String shardId = getShardId();
        ShardModelCache cache = ShardModelCache.getInstance(shardId);

        // 1. Check in-memory cache first
        byte[] cachedModel = cache.getModel(fieldInfo.name);
        if (cachedModel != null) {
            log.info("[Merge] Encoding=LeanVec (cached model) for field '{}' ({} vectors, segment={})",
                fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
            return cachedModel;
        }

        // 2. Seed counter on first merge after restart (O-19: explicit seeded flag, O-R1-14: CAS)
        if (cache.tryMarkCounterSeeded(fieldInfo.name)) {
            seedCounterFromCommittedSegments(fieldInfo, cache);
        }

        // 3. Cumulative threshold check (flush-only counting — merges don't add to counter)
        long cumulativeCount = cache.getCumulativeVectorCount(fieldInfo.name);
        int threshold = getTrainingThreshold(fieldInfo);
        if (cumulativeCount < threshold) {
            log.info("[Merge] Encoding=LVQ (cumulative {} < threshold {}) for field '{}' ({} merge vectors, segment={})",
                cumulativeCount, threshold, fieldInfo.name, totalLiveDocs, segmentWriteState.segmentInfo.name);
            return null;
        }

        // 4. Check circuit breaker — training suppressed after consecutive failures
        if (cache.isTrainingSuppressed(fieldInfo.name)) {
            log.warn("[Merge] Training suppressed by circuit breaker for field '{}' — LVQ fallback", fieldInfo.name);
            return null;
        }

        // O-9: Check KNN circuit breaker before allocating off-heap memory for training
        if (KNNSettings.isCircuitBreakerTriggered()) {
            log.warn("[Merge] KNN circuit breaker triggered, deferring LeanVec training for field '{}'", fieldInfo.name);
            return null;
        }

        // 5. C-2: Use tryLock to avoid blocking merge threads during concurrent training
        ReentrantLock lock = cache.getTrainingLock(fieldInfo.name);
        boolean acquired;
        try {
            acquired = lock.tryLock(100, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("[Merge] Interrupted while acquiring training lock for field '{}'", fieldInfo.name);
            return null;
        }
        if (!acquired) {
            log.info("[Merge] Encoding=LVQ (training lock busy, another merge is training) for field '{}' (segment={})",
                fieldInfo.name, segmentWriteState.segmentInfo.name);
            return null;
        }

        try {
            // Double-check: model may have been created while waiting for the lock
            cachedModel = cache.getModel(fieldInfo.name);
            if (cachedModel != null) {
                log.info("[Merge] Encoding=LeanVec (model appeared while waiting for lock) for field '{}' (segment={})",
                    fieldInfo.name, segmentWriteState.segmentInfo.name);
                return cachedModel;
            }

            // O-3: If merge already has >= threshold vectors, train from merge vectors directly
            // (force merge optimization — avoids unnecessary DirectoryReader I/O)
            KNNCounter.DEFERRED_TRAINING_REQUESTS.increment();
            byte[] modelBlob;
            if (totalLiveDocs >= threshold) {
                log.info(
                    "[Merge] TRAINING STARTED (merge-only path): field='{}', mergeVectors={}, threshold={}, segment={}",
                    fieldInfo.name, totalLiveDocs, threshold, segmentWriteState.segmentInfo.name
                );
                modelBlob = trainLeanVecModel(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
            } else {
                log.info(
                    "[Merge] TRAINING STARTED (DirectoryReader path): field='{}', mergeVectors={}, threshold={}, cumulativeCount={}, segment={}",
                    fieldInfo.name, totalLiveDocs, threshold, cumulativeCount, segmentWriteState.segmentInfo.name
                );
                modelBlob = trainLeanVecModelFromAllSegments(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
            }

            // K-R1-5/O-R1-18 FIX: Distinguish sentinels from real failures.
            // Only record failure (toward circuit breaker) for actual training errors.
            if (modelBlob != null && modelBlob.length > 0) {
                cache.putModel(fieldInfo.name, modelBlob);
                KNNCounter.DEFERRED_TRAINING_SUCCESS.increment();
                log.info("[Merge] TRAINING COMPLETE: Encoding=LeanVec for field '{}' (segment={})",
                    fieldInfo.name, segmentWriteState.segmentInfo.name);
                return modelBlob;
            } else if (modelBlob == TRAINING_INTERRUPTED) {
                // Transient condition — don't count as failure
                log.info("[Merge] Training interrupted (not a failure) for field '{}', using LVQ fallback", fieldInfo.name);
                return null;
            } else if (modelBlob == TRAINING_INSUFFICIENT_VECTORS) {
                // Expected condition with sparse fields — don't count as failure
                log.info("[Merge] Insufficient vectors for training field '{}', using LVQ fallback", fieldInfo.name);
                return null;
            } else {
                // Real failure (null return from training)
                cache.recordFailure(fieldInfo.name);
                KNNCounter.DEFERRED_TRAINING_ERRORS.increment();
                log.warn("[Merge] TRAINING FAILED: Encoding=LVQ fallback for field '{}' (segment={})",
                    fieldInfo.name, segmentWriteState.segmentInfo.name);
                return null;
            }
        } finally {
            lock.unlock();
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
     * Trains a LeanVec model using vectors from ALL committed segments via DirectoryReader.
     * This satisfies the SVS team's requirement that training must use at least `threshold` vectors.
     *
     * L-R1-19 fix: Streams vectors directly to OffHeapFloatVectorTransfer instead of collecting
     * all into a heap ArrayList. Keeps K-7 separation by catching DirectoryReader errors separately
     * from JNI errors — if DirectoryReader fails, we discard the partial off-heap transfer and
     * fall back to merge vectors.
     *
     * Falls back to merge-only vectors if DirectoryReader.open() fails.
     *
     * Returns TRAINING_INTERRUPTED sentinel on thread interruption, or TRAINING_INSUFFICIENT_VECTORS
     * when not enough vectors available. Caller must distinguish from real failures (K-R1-5 fix).
     */
    private byte[] trainLeanVecModelFromAllSegments(
        FieldInfo fieldInfo,
        Supplier<KNNVectorValues<?>> mergeVectorsFallback,
        int mergeVectorCount
    ) throws IOException {
        int dimension = getDimension(fieldInfo);
        int leanvecDims = getLeanVecDimensions(fieldInfo);
        if (dimension <= 0 || leanvecDims <= 0) {
            log.warn("Cannot train LeanVec: invalid dimensions (dim={}, leanvecDims={})", dimension, leanvecDims);
            return null;
        }

        int threshold = getTrainingThreshold(fieldInfo);
        int trainingCount = Math.min(threshold, MAX_TRAINING_VECTORS);

        // C-6: Use FilterDirectory.unwrap() static method (safe on Linux, inode semantics)
        Directory rawDir = FilterDirectory.unwrap(segmentWriteState.directory);

        StopWatch stopWatch = new StopWatch().start();
        boolean stopped = false;
        try {
            // Build training parameters (K-6: all 4 required params)
            String indexDescription = getIndexDescriptionFromFieldInfo(fieldInfo);
            Map<String, Object> trainParameters = new HashMap<>();
            trainParameters.put(INDEX_DESCRIPTION_PARAMETER, indexDescription);
            trainParameters.put(SPACE_TYPE, fieldInfo.attributes().getOrDefault(SPACE_TYPE, "l2"));
            trainParameters.put(INDEX_THREAD_QTY, KNNSettings.getIndexThreadQty());
            trainParameters.put(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());

            int bytesPerVector = dimension * Float.BYTES;
            boolean directoryReaderFailed = false;
            int transferred = 0;

            // L-R1-19 FIX: Stream vectors directly to off-heap. If DirectoryReader fails,
            // close the transfer and re-create for fallback path.
            try (OffHeapFloatVectorTransfer vectorTransfer = new OffHeapFloatVectorTransfer(bytesPerVector, trainingCount)) {
                try (DirectoryReader reader = DirectoryReader.open(rawDir)) {
                    // Count total available vectors for uniform sampling
                    int totalAvailable = 0;
                    for (LeafReaderContext leafCtx : reader.leaves()) {
                        FloatVectorValues fvv = leafCtx.reader().getFloatVectorValues(fieldInfo.name);
                        if (fvv != null) totalAvailable += fvv.size();
                    }

                    // R-2 FIX: Ceiling division for unbiased sampling
                    int sampleStep = Math.max(1, (totalAvailable + trainingCount - 1) / trainingCount);

                    log.info("[Merge] DirectoryReader opened: {} total vectors across {} segments for field '{}', sampleStep={}",
                        totalAvailable, reader.leaves().size(), fieldInfo.name, sampleStep);

                    int globalIdx = 0;
                    outer:
                    for (LeafReaderContext leafCtx : reader.leaves()) {
                        if (transferred >= trainingCount) break;

                        FloatVectorValues fvv = leafCtx.reader().getFloatVectorValues(fieldInfo.name);
                        if (fvv == null) continue;

                        KnnVectorValues.DocIndexIterator iter = fvv.iterator();
                        while (iter.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                            // M-5/K-14 FIX: Thread interruption check
                            if (Thread.currentThread().isInterrupted()) {
                                log.warn("[Merge] Training interrupted during DirectoryReader iteration for field '{}'", fieldInfo.name);
                                return TRAINING_INTERRUPTED;
                            }
                            if (transferred >= trainingCount) break outer;
                            if (globalIdx % sampleStep == 0) {
                                // Stream directly to off-heap — vectorValue() buffer is consumed by transfer
                                vectorTransfer.transfer(fvv.vectorValue(iter.index()).clone(), true);
                                transferred++;
                            }
                            globalIdx++;
                        }
                    }
                    log.info("[Merge] DirectoryReader streamed {} vectors from {} available for field '{}'",
                        transferred, totalAvailable, fieldInfo.name);
                } catch (IOException e) {
                    // K-7: DirectoryReader I/O error is separate from JNI errors
                    log.warn("[Merge] DirectoryReader failed for field '{}': {}", fieldInfo.name, e.getMessage(), e);
                    directoryReaderFailed = true;
                    transferred = 0;  // Discard partial off-heap data
                }

                // Phase 2: If DirectoryReader failed, fall back to merge vectors
                if (directoryReaderFailed && mergeVectorsFallback != null) {
                    // Re-create transfer (old one has partial data we can't use)
                    try (OffHeapFloatVectorTransfer fallbackTransfer = new OffHeapFloatVectorTransfer(bytesPerVector, trainingCount)) {
                        KNNVectorValues<?> mergeVectors = mergeVectorsFallback.get();
                        mergeVectors.nextDoc();
                        while (mergeVectors.docId() != DocIdSetIterator.NO_MORE_DOCS
                               && transferred < trainingCount) {
                            // M-6/K-R1-6 FIX: Clone in fallback path
                            fallbackTransfer.transfer(((float[]) mergeVectors.getVector()).clone(), true);
                            transferred++;
                            mergeVectors.nextDoc();
                        }
                        log.info("[Merge] Fallback: streamed {} vectors from merge for field '{}'",
                            transferred, fieldInfo.name);

                        // K-R1-18 FIX: Use trainingCount (not raw threshold) for sufficiency check
                        if (transferred < trainingCount) {
                            log.warn("[Merge] Only collected {} vectors, need {} for training. Deferring for field '{}'.",
                                transferred, trainingCount, fieldInfo.name);
                            return TRAINING_INSUFFICIENT_VECTORS;
                        }

                        fallbackTransfer.flush(true);
                        long vectorAddress = fallbackTransfer.getVectorAddress();

                        // M-3/K-8 FIX: AccessController.doPrivileged() around JNI call
                        byte[] modelBlob = AccessController.doPrivileged((PrivilegedAction<byte[]>) () -> {
                            return JNIService.trainIndex(trainParameters, dimension, vectorAddress, KNNEngine.FAISS);
                        });

                        if (modelBlob == null || modelBlob.length == 0) {
                            log.error("[Merge] LeanVec training produced empty model blob for field '{}'", fieldInfo.name);
                            return null;
                        }

                        long timeMs = stopWatch.stop().totalTime().millis();
                        stopped = true;
                        log.info(
                            "[Merge] LeanVec fallback training complete for field '{}' ({} vectors, {}ms, blob={}KB)",
                            fieldInfo.name, transferred, timeMs, modelBlob.length / 1024
                        );
                        return modelBlob;
                    }
                }

                // K-R1-18 FIX: Use trainingCount (not raw threshold) for sufficiency check
                if (transferred < trainingCount) {
                    log.warn("[Merge] Only streamed {} vectors, need {} for training. Deferring for field '{}'.",
                        transferred, trainingCount, fieldInfo.name);
                    return TRAINING_INSUFFICIENT_VECTORS;
                }

                vectorTransfer.flush(true);
                long vectorAddress = vectorTransfer.getVectorAddress();

                // M-3/K-8 FIX: AccessController.doPrivileged() around JNI call
                byte[] modelBlob = AccessController.doPrivileged((PrivilegedAction<byte[]>) () -> {
                    return JNIService.trainIndex(trainParameters, dimension, vectorAddress, KNNEngine.FAISS);
                });

                if (modelBlob == null || modelBlob.length == 0) {
                    log.error("[Merge] LeanVec training produced empty model blob for field '{}'", fieldInfo.name);
                    return null;
                }

                long timeMs = stopWatch.stop().totalTime().millis();
                stopped = true;
                log.info(
                    "[Merge] LeanVec DirectoryReader training complete for field '{}' ({} vectors, {}ms, blob={}KB)",
                    fieldInfo.name, transferred, timeMs, modelBlob.length / 1024
                );

                return modelBlob;
            }
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            log.error("[Merge] Failed to train LeanVec model (DirectoryReader path) for field '{}': {}",
                fieldInfo.name, e.getMessage(), e);
            return null;
        } finally {
            if (!stopped) {
                try { stopWatch.stop(); } catch (IllegalStateException ignored) { }
            }
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
