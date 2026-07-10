/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.faiss_upstream;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * REST integration tests for the experimental faiss_upstream engine ({@code hnsw_flat_panorama}). They
 * require a cluster built with {@code -Pknn.sandbox.enabled=true} (the bundled tenant jar + the isolated
 * {@code libopensearchknn_faiss_upstream}); run them via {@code :sandbox:faiss_upstream:integTest}.
 * Engine/method names are written as literals to avoid a test-time dependency on the tenant's main jar.
 */
public class FaissUpstreamPanoramaIT extends KNNRestTestCase {

    private static final String ENGINE = "faiss_upstream";
    private static final String METHOD = "hnsw_flat_panorama";
    private static final int DIMENSION = 3;
    private static final float[][] DOCS = new float[][] {
        { 1.0f, 1.0f, 1.0f },
        { 2.0f, 2.0f, 2.0f },
        { 3.0f, 3.0f, 3.0f },
        { 4.0f, 4.0f, 4.0f },
        { 5.0f, 5.0f, 5.0f } };

    @SneakyThrows
    public void testPanorama_whenBasicConfiguration_thenExactTopK() {
        // With ef_search >= the corpus size on a 5-doc index, HNSW traverses every vector, so Panorama
        // must return the exact nearest neighbors with plain L2 scores — comparable to a flat index.
        // (HNSWFlatPanorama's beam ordering is approximate in general; at this scale it is exhaustive.)
        final String indexName = "test-faissup-basic";
        final String fieldName = "test-field";
        createPanoramaIndex(indexName, fieldName, SpaceType.L2, 4, 3);

        int k = 3;
        Response response = searchKNNIndex(
            indexName,
            KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(new float[] { 1.0f, 1.0f, 1.0f })
                .k(k)
                .methodParameters(Map.of("ef_search", 64))
                .build(),
            k
        );
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());
        // Exact expected neighbors of (1,1,1): docs 0, 1, 2 in that order (bulkAddKnnDocs ids are 0-based).
        assertEquals("0", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testPanorama_withDefaults_thenSucceed() {
        final String indexName = "test-faissup-defaults";
        final String fieldName = "test-field";
        // No explicit parameters: m/ef_construction/n_levels resolve to their defaults (n_levels clamped to dim).
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, ENGINE)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, mapping);
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));

        int k = 2;
        Response response = searchKNNIndex(
            indexName,
            KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(new float[] { 2.0f, 2.0f, 2.0f })
                .k(k)
                .methodParameters(Map.of("ef_search", 64))
                .build(),
            k
        );
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
        assertEquals(k, results.size());
        assertEquals("1", results.get(0).getDocId());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testPanorama_whenInvalidEfSearch_thenRejected() {
        // ef_search must be > 0; core knows this parameter, so the query builder rejects it client-side.
        final String indexName = "test-faissup-badefsearch";
        final String fieldName = "test-field";
        createPanoramaIndex(indexName, fieldName, SpaceType.L2, 4, 3);

        // ef_search is a core-known method parameter, so the query builder rejects the invalid value
        // client-side — the same validation a REST caller gets at parse time.
        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(new float[] { 1.0f, 1.0f, 1.0f })
                .k(2)
                .methodParameters(Map.of("ef_search", 0))
                .build()
        );
        assertTrue(e.getMessage().contains("ef_search"));

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testPanorama_whenUndeclaredMethodParameter_thenRejectedAtParse() {
        // A method_parameters name no registered engine declared (KNNEngineDefinition#
        // engineSpecificQueryParameters) never reaches the engine: the parse-layer validation rejects it
        // immediately — here already in the client JVM when the query builder is built, exactly as an
        // undeclared name is rejected server-side at REST parse. This tenant declares nothing (ef_search is
        // core-known), so parsing behaves exactly as upstream.
        final String indexName = "test-faissup-unknownparam";
        final String fieldName = "test-field";
        createPanoramaIndex(indexName, fieldName, SpaceType.L2, 4, 3);

        IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(new float[] { 1.0f, 1.0f, 1.0f })
                .k(2)
                .methodParameters(Map.of("made_up_param", 7))
                .build()
        );
        assertTrue(e.getMessage().contains("made_up_param"));

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testPanorama_whenModeOnDisk_thenRejected() {
        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject("test-field")
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field("mode", "on_disk")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, ENGINE)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
        ResponseException e = expectThrows(ResponseException.class, () -> createKnnIndex("test-faissup-ondisk", mapping));
        assertTrue(EntityUtils.toString(e.getResponse().getEntity()).contains("on_disk is not supported with hnsw_flat_panorama"));
    }

    private void createPanoramaIndex(String indexName, String fieldName, SpaceType spaceType, int m, int nLevels) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, ENGINE)
            .startObject(PARAMETERS)
            .field("m", m)
            .field("n_levels", nLevels)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, builder.toString());
        bulkAddKnnDocs(indexName, fieldName, DOCS, DOCS.length);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));
    }
}
