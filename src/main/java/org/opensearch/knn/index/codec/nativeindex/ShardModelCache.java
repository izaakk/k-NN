/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-shard cache for LeanVec models. Ensures monotonic quality upgrades
 * (NONE->INITIAL->FINAL) and prevents concurrent training races.
 * Cleaned up on shard close via IndexEventListener.
 */
@Log4j2
public final class ShardModelCache {

    private static final int MAX_CONSECUTIVE_FAILURES = 3;

    /**
     * Model quality levels for two-threshold training.
     * Transitions are monotonic: NONE->INITIAL, NONE->FINAL, INITIAL->FINAL.
     */
    public enum ModelQuality {
        NONE,     // No model trained yet
        INITIAL,  // Rough model from initial_threshold vectors
        FINAL;    // Production model from final_threshold vectors

        /**
         * Returns true if this quality can be upgraded to the target quality.
         * Transitions are monotonic: NONE->INITIAL, NONE->FINAL, INITIAL->FINAL.
         */
        public boolean isUpgradeableTo(ModelQuality target) {
            return this.ordinal() < target.ordinal();
        }
    }

    /** Model blob with quality level. Use blobCopy() for external access. */
    public record CachedModel(byte[] blob, ModelQuality quality) {
        public byte[] blobCopy() {
            return blob.clone();
        }
    }

    /**
     * Composite key for per-quality failure tracking.
     * INITIAL and FINAL failures are tracked independently.
     */
    private record FailureKey(String fieldName, ModelQuality quality) {}

    /** Global registry: shardId -> ShardModelCache */
    private static final ConcurrentHashMap<String, ShardModelCache> INSTANCES = new ConcurrentHashMap<>();

    /** Per-field cached models: fieldName -> CachedModel (atomic blob + quality) */
    private final ConcurrentHashMap<String, CachedModel> cachedModels = new ConcurrentHashMap<>();

    /** Per-field training locks: fieldName -> lock */
    private final ConcurrentHashMap<String, ReentrantLock> trainingLocks = new ConcurrentHashMap<>();

    /** Per-field per-quality consecutive failure counts */
    private final ConcurrentHashMap<FailureKey, Integer> failureCounts = new ConcurrentHashMap<>();

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
            removed.cachedModels.clear();
            removed.trainingLocks.clear();
            removed.failureCounts.clear();
            log.debug("Cleaned up ShardModelCache for shard {}", shardId);
        }
    }

    /**
     * Removes cache entries whose filesystem-path key matches the given ShardId.
     * The key derived by NativeEngines990KnnVectorsWriter.getShardId() is a filesystem path
     * like "/data/nodes/0/indices/&lt;uuid&gt;/&lt;shard&gt;/index". This method matches on
     * "/indices/&lt;uuid&gt;/&lt;shard_number&gt;/" to find the right entry.
     *
     * @param shardId the OpenSearch ShardId from the shard close event
     */
    public static void removeInstancesForShard(org.opensearch.core.index.shard.ShardId shardId) {
        String pattern = "/indices/" + shardId.getIndex().getUUID() + "/" + shardId.id() + "/";
        INSTANCES.entrySet().removeIf(entry -> {
            if (entry.getKey().contains(pattern)) {
                ShardModelCache cache = entry.getValue();
                cache.cachedModels.clear();
                cache.trainingLocks.clear();
                cache.failureCounts.clear();
                log.debug("Cleaned up ShardModelCache for shard {} (key: {})", shardId, entry.getKey());
                return true;
            }
            return false;
        });
    }

    // ---- Model access ----

    public CachedModel getCachedModel(String fieldName) {
        return cachedModels.get(fieldName);
    }

    /** Returns a defensive copy of the model blob, or null if not cached. */
    public byte[] getModel(String fieldName) {
        CachedModel cached = cachedModels.get(fieldName);
        return cached != null ? cached.blobCopy() : null;
    }

    public ModelQuality getModelQuality(String fieldName) {
        CachedModel cached = cachedModels.get(fieldName);
        return cached != null ? cached.quality() : ModelQuality.NONE;
    }

    /**
     * Stores a model blob in the cache with quality monotonicity enforcement.
     * Rejects quality downgrades (FINAL->INITIAL, INITIAL->NONE).
     * Uses CAS loop for thread safety without external locking.
     *
     * @param fieldName the vector field name
     * @param modelBlob the model blob bytes
     * @param quality   the model quality level
     * @return true if the model was stored, false if rejected (downgrade)
     */
    public boolean putModel(String fieldName, byte[] modelBlob, ModelQuality quality) {
        byte[] cloned = modelBlob.clone();
        while (true) {
            CachedModel existing = cachedModels.get(fieldName);
            ModelQuality existingQuality = (existing != null) ? existing.quality() : ModelQuality.NONE;
            if (!existingQuality.isUpgradeableTo(quality) && existingQuality != quality) {
                log.warn("Rejecting model downgrade {} -> {} for field '{}'", existingQuality, quality, fieldName);
                return false;
            }
            CachedModel newModel = new CachedModel(cloned, quality);
            if (existing == null) {
                if (cachedModels.putIfAbsent(fieldName, newModel) == null) {
                    clearFailuresOnSuccess(fieldName, quality);
                    return true;
                }
                // Lost race — retry
            } else {
                if (cachedModels.replace(fieldName, existing, newModel)) {
                    clearFailuresOnSuccess(fieldName, quality);
                    return true;
                }
                // Lost race — retry
            }
        }
    }

    public boolean hasModel(String fieldName) {
        return cachedModels.containsKey(fieldName);
    }

    // ---- Training locks ----

    public ReentrantLock getTrainingLock(String fieldName) {
        return trainingLocks.computeIfAbsent(fieldName, k -> new ReentrantLock());
    }

    // ---- Per-quality circuit breaker ----

    public void recordFailure(String fieldName, ModelQuality quality) {
        FailureKey key = new FailureKey(fieldName, quality);
        int count = failureCounts.merge(key, 1, Integer::sum);
        if (count >= MAX_CONSECUTIVE_FAILURES) {
            log.warn("Circuit breaker: {} consecutive {} training failures for field '{}'. "
                + "Suppressing further {} attempts.",
                count, quality, fieldName, quality);
        }
    }

    public boolean isTrainingSuppressed(String fieldName, ModelQuality quality) {
        return failureCounts.getOrDefault(new FailureKey(fieldName, quality), 0) >= MAX_CONSECUTIVE_FAILURES;
    }

    /**
     * Clears failure counts on successful training. Clears both the stored quality's
     * failures and any lower-quality failures.
     */
    private void clearFailuresOnSuccess(String fieldName, ModelQuality quality) {
        failureCounts.remove(new FailureKey(fieldName, quality));
        // When upgrading to FINAL, also clear stale INITIAL failures
        if (quality == ModelQuality.FINAL) {
            failureCounts.remove(new FailureKey(fieldName, ModelQuality.INITIAL));
        }
    }

}
