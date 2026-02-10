/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;

/**
 * Reads LeanVec model blobs from Lucene segment files (.knnlvm).
 *
 * File format mirrors {@link LeanVecModelWriter}.
 */
@Log4j2
public final class LeanVecModelReader {

    /**
     * Reads a LeanVec model blob for a given field from a segment.
     *
     * @param segmentReadState the segment read state
     * @param fieldName        the field name to look up
     * @return the model blob bytes, or null if no model exists for this field
     */
    public static byte[] read(SegmentReadState segmentReadState, String fieldName) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            segmentReadState.segmentInfo.name,
            segmentReadState.segmentSuffix,
            KNNConstants.LEANVEC_MODEL_FILE_SUFFIX
        );

        // Check if the file exists in this segment
        if (!fileExistsInSegment(segmentReadState.directory, fileName)) {
            return null;
        }

        int fieldNumber = segmentReadState.fieldInfos.fieldInfo(fieldName).getFieldNumber();

        try (IndexInput input = segmentReadState.directory.openInput(fileName, IOContext.READONCE)) {
            CodecUtil.checkIndexHeader(
                input,
                LeanVecModelWriter.LEANVEC_MODEL_DATA_CODEC,
                LeanVecModelWriter.LEANVEC_MODEL_VERSION,
                LeanVecModelWriter.LEANVEC_MODEL_VERSION,
                segmentReadState.segmentInfo.getId(),
                segmentReadState.segmentSuffix
            );

            // Navigate to the index section (same layout as KNN990QuantizationStateReader)
            long footerStart = input.length() - CodecUtil.footerLength();
            long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
            input.seek(markerAndIndexPosition);
            long indexStartPosition = input.readLong();
            input.seek(indexStartPosition);

            int numFields = input.readInt();
            long position = -1;
            int length = 0;

            for (int i = 0; i < numFields; i++) {
                int tempFieldNumber = input.readInt();
                int tempLength = input.readInt();
                long tempPosition = input.readVLong();
                if (tempFieldNumber == fieldNumber) {
                    position = tempPosition;
                    length = tempLength;
                    break;
                }
            }

            if (position == -1 || length == 0) {
                return null;
            }

            input.seek(position);
            byte[] modelBlob = new byte[length];
            input.readBytes(modelBlob, 0, length);

            CodecUtil.retrieveChecksum(input);

            return modelBlob;
        }
    }

    /**
     * Reads a LeanVec model blob from a merge input segment.
     *
     * @param dir         the directory containing the segment
     * @param segmentInfo the segment commit info
     * @param fieldName   the field name
     * @param fieldNumber the field number
     * @return the model blob, or null if not found
     */
    public static byte[] readFromSegment(Directory dir, SegmentCommitInfo segmentInfo, String fieldName, int fieldNumber) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            segmentInfo.info.name,
            "",
            KNNConstants.LEANVEC_MODEL_FILE_SUFFIX
        );

        if (!fileExistsInSegment(dir, fileName)) {
            return null;
        }

        try (IndexInput input = dir.openInput(fileName, IOContext.READONCE)) {
            CodecUtil.checkIndexHeader(
                input,
                LeanVecModelWriter.LEANVEC_MODEL_DATA_CODEC,
                LeanVecModelWriter.LEANVEC_MODEL_VERSION,
                LeanVecModelWriter.LEANVEC_MODEL_VERSION,
                segmentInfo.info.getId(),
                ""
            );

            long footerStart = input.length() - CodecUtil.footerLength();
            long markerAndIndexPosition = footerStart - Integer.BYTES - Long.BYTES;
            input.seek(markerAndIndexPosition);
            long indexStartPosition = input.readLong();
            input.seek(indexStartPosition);

            int numFields = input.readInt();
            long position = -1;
            int length = 0;

            for (int i = 0; i < numFields; i++) {
                int tempFieldNumber = input.readInt();
                int tempLength = input.readInt();
                long tempPosition = input.readVLong();
                if (tempFieldNumber == fieldNumber) {
                    position = tempPosition;
                    length = tempLength;
                    break;
                }
            }

            if (position == -1 || length == 0) {
                return null;
            }

            input.seek(position);
            byte[] modelBlob = new byte[length];
            input.readBytes(modelBlob, 0, length);

            return modelBlob;
        }
    }

    private static boolean fileExistsInSegment(Directory dir, String fileName) {
        try {
            dir.openInput(fileName, IOContext.READONCE).close();
            return true;
        } catch (IOException e) {
            return false;
        }
    }
}
