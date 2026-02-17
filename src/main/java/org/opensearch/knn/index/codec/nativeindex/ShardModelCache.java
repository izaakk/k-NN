/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-memory cache for LeanVec model blobs, scoped to a shard.
 *
 * Provides:
 * - O(1) model lookup during flush (avoids re-reading segment files)
 * - Atomic model+quality storage via CachedModel record
 * - Monotonic quality upgrades: NONE to INITIAL to FINAL
 * - Per-field per-quality failure circuit breaker
 * - Per-field training locks (prevents concurrent training races)
 *
 * Lifecycle: created per-shard, cleaned up via IndexEventListener.afterIndexShardClosed().
 */
@Log4j2
public final class ShardModelCache {

    private static final int MAX_CONSECUTIVE_FAILURES = 3;

    /**
     * Model quality levels for two-threshold training.
     * Transitions are monotonic: NONE to INITIAL to FINAL.
     */
    public enum ModelQuality {
        NONE,
        INITIAL,
        FINAL;

        public boolean isUpgradeableTo(ModelQuality target) {
            return this.ordinal() < target.ordinal();
        }
    }

    /**
     * Immutable record wrapping a model blob with its quality level.
     * Use {@link #blobCopy()} when passing the blob outside the cache to prevent mutation.
     */
    public record CachedModel(byte[] blob, ModelQuality quality) {
        /**
         * Returns a defensive copy of the blob.
         */
        public byte[] blobCopy() {
            return blob.clone();
        }
    }

    /** Composite key for per-quality failure tracking. INITIAL and FINAL failures are tracked independently. */
    private record FailureKey(String fieldName, ModelQuality quality) {}

    /** Global registry: shardId -> ShardModelCache */
    private static final ConcurrentHashMap<String, ShardModelCache> INSTANCES = new ConcurrentHashMap<>();

    /** Per-field cached models: fieldName -> CachedModel (atomic blob + quality) */
    private final ConcurrentHashMap<String, CachedModel> cachedModels = new ConcurrentHashMap<>();

    /** Per-field training locks: fieldName -> lock */
    private final ConcurrentHashMap<String, ReentrantLock> trainingLocks = new ConcurrentHashMap<>();

    /** Per-field per-quality consecutive failure counts */
    private final ConcurrentHashMap<FailureKey, Integer> failureCounts = new ConcurrentHashMap<>();

    /** Per-field cumulative vector counts: fieldName -> count (incremented on flush only, not merge) */
    private final ConcurrentHashMap<String, AtomicLong> cumulativeVectorCounts = new ConcurrentHashMap<>();

    /** Per-field counter-seeded flags to handle flush-before-merge race */
    private final ConcurrentHashMap<String, AtomicBoolean> counterSeeded = new ConcurrentHashMap<>();

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
            removed.cumulativeVectorCounts.clear();
            removed.counterSeeded.clear();
            log.debug("Cleaned up ShardModelCache for shard {}", shardId);
        }
    }

    /**
     * Removes cache entries whose filesystem-path key matches the given ShardId.
     * The key derived by NativeEngines990KnnVectorsWriter.getShardId() is a filesystem path
     * like "/data/nodes/0/indices/&lt;uuid&gt;/&lt;shard&gt;/index". This method matches on
     * "/indices/&lt;uuid&gt;/&lt;shard_number&gt;/" to find the right entry.
     */
    public static void removeInstancesForShard(org.opensearch.core.index.shard.ShardId shardId) {
        String pattern = "/indices/" + shardId.getIndex().getUUID() + "/" + shardId.id() + "/";
        INSTANCES.entrySet().removeIf(entry -> {
            if (entry.getKey().contains(pattern)) {
                ShardModelCache cache = entry.getValue();
                cache.cachedModels.clear();
                cache.trainingLocks.clear();
                cache.failureCounts.clear();
                cache.cumulativeVectorCounts.clear();
                cache.counterSeeded.clear();
                log.debug("Cleaned up ShardModelCache for shard {} (key: {})", shardId, entry.getKey());
                return true;
            }
            return false;
        });
    }

    // ---- Model access ----

    /**
     * Returns the cached model entry, or null if not cached.
     */
    public CachedModel getCachedModel(String fieldName) {
        return cachedModels.get(fieldName);
    }

    /**
     * Returns a defensive copy of the cached model blob, or null if not cached.
     */
    public byte[] getModel(String fieldName) {
        CachedModel cached = cachedModels.get(fieldName);
        return cached != null ? cached.blobCopy() : null;
    }

    /**
     * Returns the model quality for a field, or NONE if no model cached.
     */
    public ModelQuality getModelQuality(String fieldName) {
        CachedModel cached = cachedModels.get(fieldName);
        return cached != null ? cached.quality() : ModelQuality.NONE;
    }

    /**
     * Stores a model blob in the cache. Rejects quality downgrades.
     * Uses CAS loop for thread safety without external locking.
     *
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

    /**
     * Checks if a model exists for the given field.
     */
    public boolean hasModel(String fieldName) {
        return cachedModels.containsKey(fieldName);
    }

    // ---- Training locks ----

    /**
     * Gets the per-field training lock.
     */
    public ReentrantLock getTrainingLock(String fieldName) {
        return trainingLocks.computeIfAbsent(fieldName, k -> new ReentrantLock());
    }

    // ---- Per-quality circuit breaker ----

    /**
     * Records a training failure for a specific quality level.
     */
    public void recordFailure(String fieldName, ModelQuality quality) {
        FailureKey key = new FailureKey(fieldName, quality);
        int count = failureCounts.merge(key, 1, Integer::sum);
        if (count >= MAX_CONSECUTIVE_FAILURES) {
            log.warn("Circuit breaker: {} consecutive {} training failures for field '{}'. "
                + "Suppressing further {} attempts.",
                count, quality, fieldName, quality);
        }
    }

    /**
     * Checks if training is suppressed for a specific quality level.
     */
    public boolean isTrainingSuppressed(String fieldName, ModelQuality quality) {
        return failureCounts.getOrDefault(new FailureKey(fieldName, quality), 0) >= MAX_CONSECUTIVE_FAILURES;
    }

    /**
     * Clears failure counts on successful training and any lower-quality failures.
     */
    private void clearFailuresOnSuccess(String fieldName, ModelQuality quality) {
        failureCounts.remove(new FailureKey(fieldName, quality));
        // When upgrading to FINAL, also clear stale INITIAL failures
        if (quality == ModelQuality.FINAL) {
            failureCounts.remove(new FailureKey(fieldName, ModelQuality.INITIAL));
        }
    }

    // ---- Cumulative vector counting (for cumulative-threshold training) ----

    /**
     * Adds vectors to the cumulative count for a field. Called from flush() only
     * to avoid double-counting from merges.
     *
     * @return the new cumulative count
     */
    public long addVectors(String fieldName, long count) {
        return cumulativeVectorCounts.computeIfAbsent(fieldName, k -> new AtomicLong(0))
            .addAndGet(count);
    }

    /**
     * Returns the current cumulative vector count for a field, or 0 if not tracked.
     */
    public long getCumulativeVectorCount(String fieldName) {
        AtomicLong counter = cumulativeVectorCounts.get(fieldName);
        return counter != null ? counter.get() : 0;
    }

    /**
     * Seeds the counter from committed segment metadata on first merge after restart.
     */
    public void seedVectorCount(String fieldName, long count) {
        cumulativeVectorCounts.computeIfAbsent(fieldName, k -> new AtomicLong(0))
            .addAndGet(count);
    }

    /**
     * Checks whether the cumulative counter has been seeded for a field.
     */
    public boolean isCounterSeeded(String fieldName) {
        AtomicBoolean flag = counterSeeded.get(fieldName);
        return flag != null && flag.get();
    }

    /**
     * Atomically marks the counter as seeded using CAS.
     *
     * @return true if this call won the race and the caller should seed; false if already seeded
     */
    public boolean tryMarkCounterSeeded(String fieldName) {
        return counterSeeded.computeIfAbsent(fieldName, k -> new AtomicBoolean(false))
            .compareAndSet(false, true);
    }
}
