/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * REST integration tests for the experimental {@code svs_vamana} method, requiring a node with the
 * SVS-enabled plugin (sandbox jar bundled, native lib built with {@code FAISS_ENABLE_SVS=ON}); runs in CI on
 * Intel runners. LeanVec is deferred to the training follow-up. SVS param names are written as literals to
 * avoid a test-time dependency on the sandbox module.
 */
public class FaissSVSVamanaIT extends KNNRestTestCase {

    private static final String SVS_VAMANA = "svs_vamana";
    private static final int DIMENSION = 3;
    private static final float[][] DOCS = new float[][] { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };

    @SneakyThrows
    public void testSVSVamana_whenBasicConfiguration_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-basic", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS).field("degree", 64).endObject();
        });
    }

    @SneakyThrows
    public void testSVSVamana_withSqFp16Encoder_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-sq-fp16", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS)
                .field("degree", 64)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "sq")
                .startObject(PARAMETERS)
                .field("type", "fp16")
                .endObject()
                .endObject()
                .endObject();
        });
    }

    @SneakyThrows
    public void testSVSVamana_withLvqEncoder_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-lvq", SpaceType.L2, builder -> {
            builder.startObject(PARAMETERS)
                .field("degree", 64)
                .startObject(METHOD_ENCODER_PARAMETER)
                .field(NAME, "lvq")
                .startObject(PARAMETERS)
                .field("primary_bits", 4)
                .field("residual_bits", 4)
                .endObject()
                .endObject()
                .endObject();
        });
    }

    // Asserts the SVS search context accepts a query-time search_window_size method parameter.
    @SneakyThrows
    public void testSVSVamana_withSearchWindowSizeMethodParameter_thenSucceed() {
        final String indexName = "test-svs-vamana-sw-param";
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field("degree", 64)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, "lvq")
            .startObject(PARAMETERS)
            .field("primary_bits", 4)
            .field("residual_bits", 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));

        int k = 2;
        KNNQueryBuilder query = KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(new float[] { 1.0f, 1.0f, 1.0f })
            .k(k)
            .methodParameters(Map.of("search_window_size", 64))
            .build();
        Response response = searchKNNIndex(indexName, query, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_withCosinesimil_thenSucceed() {
        runIndexSearchRoundtrip("test-svs-vamana-cosine", SpaceType.COSINESIMIL, builder -> {
            builder.startObject(PARAMETERS).field("degree", 64).endObject();
        });
    }

    @FunctionalInterface
    private interface ParamsWriter {
        void write(XContentBuilder builder) throws Exception;
    }

    private void runIndexSearchRoundtrip(String indexName, SpaceType spaceType, ParamsWriter paramsWriter) throws Exception {
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName());
        paramsWriter.write(builder);
        builder.endObject().endObject().endObject().endObject();

        createKnnIndex(indexName, builder.toString());
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));

        int k = 2;
        Response response = searchKNNIndex(indexName, new KNNQueryBuilder(fieldName, new float[] { 1.0f, 1.0f, 1.0f }, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testSVSVamana_whenModeOnDisk_thenRejected() {
        final String fieldName = "test-field";
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        ResponseException e = expectThrows(ResponseException.class, () -> createKnnIndex("test-svs-vamana-ondisk", builder.toString()));
        assertTrue(EntityUtils.toString(e.getResponse().getEntity()).contains("on_disk is not supported with svs_vamana"));
    }
}
