/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-memory cache for LeanVec model blobs, scoped to a shard.
 *
 * Provides:
 * - O(1) model lookup during flush (avoids re-reading segment files)
 * - Per-field training locks (prevents concurrent training races)
 * - Circuit breaker: after MAX_CONSECUTIVE_FAILURES consecutive training failures,
 *   further training attempts are suppressed for this field
 *
 * Lifecycle: created per-shard, cleaned up via KNNPlugin.onIndexModule() →
 * IndexEventListener.afterIndexShardClosed().
 */
@Log4j2
public final class ShardModelCache {

    private static final int MAX_CONSECUTIVE_FAILURES = 3;

    /** Global registry: shardId → ShardModelCache */
    private static final ConcurrentHashMap<String, ShardModelCache> INSTANCES = new ConcurrentHashMap<>();

    /** Per-field model blobs: fieldName → modelBlob */
    private final ConcurrentHashMap<String, byte[]> models = new ConcurrentHashMap<>();

    /** Per-field training locks: fieldName → lock */
    private final ConcurrentHashMap<String, ReentrantLock> trainingLocks = new ConcurrentHashMap<>();

    /** Per-field consecutive failure counts: fieldName → count */
    private final ConcurrentHashMap<String, Integer> failureCounts = new ConcurrentHashMap<>();

    /**
     * Gets (or creates) the cache for a given shard.
     *
     * @param shardId unique shard identifier (e.g., "[index][0]")
     * @return the ShardModelCache for this shard
     */
    public static ShardModelCache getInstance(String shardId) {
        return INSTANCES.computeIfAbsent(shardId, k -> new ShardModelCache());
    }

    /**
     * Removes and cleans up the cache for a shard by exact key (called on shard close).
     */
    public static void removeInstance(String shardId) {
        ShardModelCache removed = INSTANCES.remove(shardId);
        if (removed != null) {
            removed.models.clear();
            removed.trainingLocks.clear();
            removed.failureCounts.clear();
            log.debug("Cleaned up ShardModelCache for shard {}", shardId);
        }
    }

    /**
     * Removes cache entries whose filesystem-path key matches the given ShardId.
     * The key derived by NativeEngines990KnnVectorsWriter.getShardId() is a filesystem path
     * like "/data/nodes/0/indices/&lt;uuid&gt;/&lt;shard&gt;/index". This method matches on
     * "/indices/&lt;uuid&gt;/&lt;shard_number&gt;/" to find the right entry (C-1 fix).
     *
     * @param shardId the OpenSearch ShardId from the shard close event
     */
    public static void removeInstancesForShard(org.opensearch.core.index.shard.ShardId shardId) {
        String pattern = "/indices/" + shardId.getIndex().getUUID() + "/" + shardId.id() + "/";
        INSTANCES.entrySet().removeIf(entry -> {
            if (entry.getKey().contains(pattern)) {
                ShardModelCache cache = entry.getValue();
                cache.models.clear();
                cache.trainingLocks.clear();
                cache.failureCounts.clear();
                log.debug("Cleaned up ShardModelCache for shard {} (key: {})", shardId, entry.getKey());
                return true;
            }
            return false;
        });
    }

    /**
     * Gets a cached model blob for a field.
     *
     * @param fieldName the vector field name
     * @return the model blob, or null if not cached
     */
    public byte[] getModel(String fieldName) {
        return models.get(fieldName);
    }

    /**
     * Stores a model blob in the cache.
     *
     * @param fieldName the vector field name
     * @param modelBlob the model blob bytes
     */
    public void putModel(String fieldName, byte[] modelBlob) {
        models.put(fieldName, modelBlob);
        failureCounts.remove(fieldName);
    }

    /**
     * Gets the per-field training lock.
     */
    public ReentrantLock getTrainingLock(String fieldName) {
        return trainingLocks.computeIfAbsent(fieldName, k -> new ReentrantLock());
    }

    /**
     * Records a training failure for circuit breaker logic (W-2 fix: use merge return value).
     */
    public void recordFailure(String fieldName) {
        int count = failureCounts.merge(fieldName, 1, Integer::sum);
        if (count >= MAX_CONSECUTIVE_FAILURES) {
            log.warn("Circuit breaker: {} consecutive training failures for field {}. Suppressing further attempts.", count, fieldName);
        }
    }

    /**
     * Checks if training is suppressed by the circuit breaker.
     */
    public boolean isTrainingSuppressed(String fieldName) {
        return failureCounts.getOrDefault(fieldName, 0) >= MAX_CONSECUTIVE_FAILURES;
    }

    /**
     * Checks if a model exists for the given field.
     */
    public boolean hasModel(String fieldName) {
        return models.containsKey(fieldName);
    }
}
