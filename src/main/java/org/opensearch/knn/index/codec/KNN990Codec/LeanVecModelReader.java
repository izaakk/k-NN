/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.nativeindex.ShardModelCache;

import java.io.IOException;

/**
 * Reads LeanVec model blobs from Lucene segment files (.knnlvm).
 * Supports both version 0 (no quality) and version 1 (with quality ordinal).
 */
@Log4j2
public final class LeanVecModelReader {

    /** Result of reading a model: blob bytes + quality level. */
    public record ModelReadResult(byte[] blob, ShardModelCache.ModelQuality quality) {}

    public static ModelReadResult readFromSegment(
        Directory dir, SegmentCommitInfo segmentInfo, String fieldName, int fieldNumber
    ) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            segmentInfo.info.name,
            "",
            KNNConstants.LEANVEC_MODEL_FILE_SUFFIX
        );

        if (!fileExistsInSegment(dir, fileName)) {
            return null;
        }

        return readModelFromFile(dir, fileName, segmentInfo.info.getId(), "", fieldNumber);
    }

    /**
     * Shared implementation for reading a model from a .knnlvm file.
     * Handles both version 0 (no quality) and version 1 (with quality ordinal).
     */
    private static ModelReadResult readModelFromFile(
        Directory dir, String fileName, byte[] segmentId, String segmentSuffix, int fieldNumber
    ) throws IOException {
        try (IndexInput input = dir.openInput(fileName, IOContext.DEFAULT)) {
            int version = CodecUtil.checkIndexHeader(
                input,
                LeanVecModelWriter.LEANVEC_MODEL_DATA_CODEC,
                LeanVecModelWriter.LEANVEC_MODEL_MIN_VERSION,
                LeanVecModelWriter.LEANVEC_MODEL_VERSION,
                segmentId,
                segmentSuffix
            );

            long footerStart = input.length() - CodecUtil.footerLength();
            long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
            input.seek(markerAndIndexPosition);
            long indexStartPosition = input.readLong();
            input.seek(indexStartPosition);

            int numFields = input.readInt();
            long position = -1;
            int length = 0;
            int qualityOrdinal = ShardModelCache.ModelQuality.INITIAL.ordinal(); // default for v0

            for (int i = 0; i < numFields; i++) {
                int tempFieldNumber = input.readInt();
                int tempLength = input.readInt();
                int tempQuality = version >= 1 ? input.readInt() : ShardModelCache.ModelQuality.INITIAL.ordinal();
                long tempPosition = input.readVLong();
                if (tempFieldNumber == fieldNumber) {
                    position = tempPosition;
                    length = tempLength;
                    qualityOrdinal = tempQuality;
                    break;
                }
            }

            if (position == -1 || length == 0) {
                return null;
            }

            input.seek(position);
            byte[] modelBlob = new byte[length];
            input.readBytes(modelBlob, 0, length);

            CodecUtil.checksumEntireFile(input);

            ShardModelCache.ModelQuality[] values = ShardModelCache.ModelQuality.values();
            ShardModelCache.ModelQuality quality = (qualityOrdinal >= 0 && qualityOrdinal < values.length)
                ? values[qualityOrdinal]
                : ShardModelCache.ModelQuality.INITIAL;

            return new ModelReadResult(modelBlob, quality);
        }
    }

    private static boolean fileExistsInSegment(Directory dir, String fileName) throws IOException {
        for (String file : dir.listAll()) {
            if (file.equals(fileName)) return true;
        }
        return false;
    }
}
