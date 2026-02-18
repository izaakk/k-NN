/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.AllArgsConstructor;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.KNNConstants;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes LeanVec model blobs as Lucene segment files (.knnlvm).
 *
 * File format (version 1):
 * Header (CodecUtil index header)
 * Model1 blob bytes
 * Model2 blob bytes
 * Number of models (int)
 * Model1 field number (int)
 * Model1 blob length (int)
 * Model1 quality ordinal (int)   -- added in version 1
 * Model1 position (vlong)
 * Model2 field number (int)
 * Model2 blob length (int)
 * Model2 quality ordinal (int)   -- added in version 1
 * Model2 position (vlong)
 * Position of index section (long)
 * -1 marker (int)
 * Footer (CodecUtil footer with CRC32)
 *
 * Follows the same pattern as {@link KNN990QuantizationStateWriter}.
 */
public final class LeanVecModelWriter implements Closeable {

    static final String LEANVEC_MODEL_DATA_CODEC = "NativeEngines990KnnVectorsFormatLeanVecModelData";
    static final int LEANVEC_MODEL_VERSION = 1;
    static final int LEANVEC_MODEL_MIN_VERSION = 0;

    private final IndexOutput output;
    private final List<FieldModel> fieldModels = new ArrayList<>();

    public LeanVecModelWriter(SegmentWriteState segmentWriteState) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            KNNConstants.LEANVEC_MODEL_FILE_SUFFIX
        );
        output = segmentWriteState.directory.createOutput(fileName, segmentWriteState.context);
    }

    public void writeHeader(SegmentWriteState segmentWriteState) throws IOException {
        CodecUtil.writeIndexHeader(
            output,
            LEANVEC_MODEL_DATA_CODEC,
            LEANVEC_MODEL_VERSION,
            segmentWriteState.segmentInfo.getId(),
            segmentWriteState.segmentSuffix
        );
    }

    public void writeModel(int fieldNumber, byte[] modelBlob, int qualityOrdinal) throws IOException {
        long position = output.getFilePointer();
        output.writeBytes(modelBlob, modelBlob.length);
        fieldModels.add(new FieldModel(fieldNumber, modelBlob.length, position, qualityOrdinal));
    }

    public void writeFooter() throws IOException {
        long indexStartPosition = output.getFilePointer();
        output.writeInt(fieldModels.size());
        for (FieldModel fm : fieldModels) {
            output.writeInt(fm.fieldNumber);
            output.writeInt(fm.blobLength);
            output.writeInt(fm.qualityOrdinal);
            output.writeVLong(fm.position);
        }
        output.writeLong(indexStartPosition);
        output.writeInt(-1);
        CodecUtil.writeFooter(output);
    }

    @Override
    public void close() throws IOException {
        if (output != null) {
            output.close();
        }
    }

    @AllArgsConstructor
    private static class FieldModel {
        final int fieldNumber;
        final int blobLength;
        final long position;
        final int qualityOrdinal;
    }
}
