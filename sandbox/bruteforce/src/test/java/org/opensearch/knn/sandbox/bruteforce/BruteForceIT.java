/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.bruteforce;

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

/**
 * End to end tests for the example engine against a real cluster: exact results, the scan_limit
 * query parameter, and rejection of invalid or undeclared parameters.
 */
public class BruteForceIT extends KNNRestTestCase {

    private static final String FIELD_NAME = "test-field";
    private static final int DIMENSION = 2;
    // Doc i gets bulk id i and sits at (i, i).
    private static final float[][] DOCS = new float[][] { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 5 } };
    private static final float[] QUERY = new float[] { 4.2f, 4.2f };

    @SneakyThrows
    public void testBruteForce_exactResults() {
        final String indexName = "test-bruteforce-exact";
        createBruteForceIndex(indexName);

        final List<KNNResult> results = search(indexName, 3, null);
        // Nearest to (4.2, 4.2) by L2: docs 4, 5, 3, in that order.
        assertEquals(List.of("4", "5", "3"), results.stream().map(KNNResult::getDocId).toList());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testBruteForce_scanLimitChangesResults() {
        final String indexName = "test-bruteforce-scan-limit";
        createBruteForceIndex(indexName);

        // Only the first two vectors get scanned, so the nearest are docs 1 and 0.
        final List<KNNResult> results = search(indexName, 2, Map.of(BruteForceLibrary.SCAN_LIMIT, 2));
        assertEquals(List.of("1", "0"), results.stream().map(KNNResult::getDocId).toList());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testBruteForce_invalidScanLimitRejected() {
        final String indexName = "test-bruteforce-bad-limit";
        createBruteForceIndex(indexName);

        expectThrows(ResponseException.class, () -> search(indexName, 2, Map.of(BruteForceLibrary.SCAN_LIMIT, 0)));

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testBruteForce_undeclaredParameterRejected() {
        final String indexName = "test-bruteforce-undeclared";
        createBruteForceIndex(indexName);

        expectThrows(ResponseException.class, () -> search(indexName, 2, Map.of("ef_search", 100)));

        deleteKNNIndex(indexName);
    }

    private void createBruteForceIndex(String indexName) throws Exception {
        final XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, BruteForceLibrary.METHOD_SCAN)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, BruteForceLibrary.ENGINE_NAME)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, builder.toString());
        bulkAddKnnDocs(indexName, FIELD_NAME, DOCS, DOCS.length);
        // scan_limit applies per segment, so the scan-limit assertions need all docs in one segment in
        // insertion order.
        forceMergeKnnIndex(indexName);
        refreshAllNonSystemIndices();
        assertEquals(DOCS.length, getDocCount(indexName));
    }

    private List<KNNResult> search(String indexName, int k, Map<String, ?> methodParameters) throws Exception {
        final KNNQueryBuilder.Builder query = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY).k(k);
        if (methodParameters != null) {
            query.methodParameters(methodParameters);
        }
        final Response response = searchKNNIndex(indexName, query.build(), k);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
    }
}
