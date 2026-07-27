/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;

/**
 * Nested-field (multi-vector) search ITs for the experimental {@code svs} engine, mirroring the core
 * {@code NestedSearchIT} scenarios for faiss. Nested SVS search runs through the SVS runtime's
 * {@code IDGrouper} (one best child per parent selected during graph search), so these tests assert the same
 * k-distinct-parents guarantee the patched-faiss engines provide. Requires a node built with
 * {@code -Pknn.sandbox.enabled=true}.
 */
public class FaissSVSVamanaNestedIT extends KNNRestTestCase {

    private static final String SVS_ENGINE = "svs";
    private static final String SVS_VAMANA = "svs_vamana";
    private static final String INDEX_NAME = "test-svs-nested-search";
    private static final String FIELD_NAME_NESTED = "test_nested";
    private static final String FIELD_NAME_VECTOR = "test_vector";
    private static final String FIELD_NAME_PARKING = "parking";
    private static final String FIELD_VALUE_TRUE = "true";
    private static final String FIELD_VALUE_FALSE = "false";
    private static final String PROPERTIES_FIELD = "properties";
    // Core's test-only o.o.knn.common.Constants is not on the sandbox test classpath; inline the two literals.
    private static final String FIELD_FILTER = "filter";
    private static final String FIELD_TERM = "term";

    /**
     * Mirror of {@code NestedSearchIT#testNestedSearchWithFaiss_whenKIsTwo_thenReturnTwoResults}: every
     * document has two identical child vectors, so k=2 must return the two distinct nearest parents, not the
     * two children of the nearest parent.
     */
    @SneakyThrows
    public void testNestedSearch_whenKIsTwo_thenReturnTwoResults() {
        createNestedSvsIndex(2);

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * The adversarial case a plain top-k cannot solve: one document's many near-identical children crowd out
     * every other document. k=3 must still return three distinct parents, led by the crowding document.
     */
    @SneakyThrows
    public void testNestedSearch_whenOneDocCrowdsTopK_thenReturnDistinctParents() {
        createNestedSvsIndex(2);

        // Doc 0: ten children all right next to the query point.
        Float[][] crowdingChildren = new Float[10][];
        for (int c = 0; c < crowdingChildren.length; c++) {
            crowdingChildren[c] = new Float[] { 0.001f * c, 0.001f * c };
        }
        addKnnDoc(INDEX_NAME, "0", NestedKnnDocBuilder.create(FIELD_NAME_NESTED).addVectors(FIELD_NAME_VECTOR, crowdingChildren).build());
        // Docs 1-5: one child each, at increasing distance.
        for (int i = 1; i <= 5; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 0f, 0f };
        Response response = queryNestedField(INDEX_NAME, 3, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(3, parseHits(entity));
        List<String> docIds = parseIds(entity);
        assertEquals("0", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals("2", docIds.get(2));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Mirror of {@code NestedSearchIT#testNestedSearchWithFaiss_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed}.
     */
    @SneakyThrows
    public void testNestedSearch_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed() {
        createNestedSvsIndex(2);

        String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .addVectors(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 2f, 2f })
            .build();
        addKnnDoc(INDEX_NAME, "1", doc);
        flushIndex(INDEX_NAME);

        addKnnDoc(INDEX_NAME, "2", "{}");
        flushIndex(INDEX_NAME);

        refreshIndex(INDEX_NAME);

        Float[] queryVector = { 1f, 1f };
        Response response = queryNestedField(INDEX_NAME, 1, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(1, parseHits(entity));
        assertEquals("1", parseIds(entity).get(0));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Mirror of {@code NestedSearchIT#testNestedSearchWithFaiss_whenDoingExactSearch_thenReturnCorrectResults}:
     * a filtered nested query pushed onto the exact-search path (threshold above the filter cardinality)
     * exercises the engine-agnostic {@code ExactSearcher} nested handling for SVS segments.
     */
    @SneakyThrows
    public void testNestedSearch_whenDoingExactSearch_thenReturnCorrectResults() {
        createNestedSvsIndex(3);

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Make it an exact search by setting the threshold larger than the size of filteredIds (6).
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f };
        Response response = queryNestedField(INDEX_NAME, 3, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, null);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Mirror of {@code NestedSearchIT#testNestedWithFaiss_whenFilter_whenDoRadialSearch_thenReturnCorrectResults}:
     * filtered nested radial search (L2 min_score converts to a positive radius) returns the best child per
     * matching parent within the radius.
     */
    @SneakyThrows
    public void testNestedSearch_whenFilter_whenDoRadialSearch_thenReturnCorrectResults() {
        createNestedSvsIndex(3);

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Keep the filtered radial query on the native ANN path: core's filtered-exact-search fallback for
        // radial still gates on engine == FAISS and rejects registered engines (known core follow-up,
        // documented in sandbox/svs/README.md).
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 0));

        Float[] queryVector = { 3f, 3f, 3f };
        Float minScore = 0.00001f;
        Response response = queryNestedField(INDEX_NAME, null, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, minScore);

        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));

        deleteKNNIndex(INDEX_NAME);
    }

    // ------------------------------------------------------------------ helpers

    private void createNestedSvsIndex(final int dimension) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, SVS_VAMANA)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNN_ENGINE, SVS_ENGINE)
            .startObject(PARAMETERS)
            .field("degree", 64)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());
    }

    private Response queryNestedField(final String index, final Integer k, final Object[] vector) throws IOException {
        return queryNestedField(index, k, vector, null, null, null);
    }

    private Response queryNestedField(
        final String index,
        final Integer k,
        final Object[] vector,
        final String filterName,
        final String filterValue,
        final Float minScore
    ) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, FIELD_NAME_NESTED);
        builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
        builder.field(VECTOR, vector);
        if (minScore != null) {
            builder.field(MIN_SCORE, minScore);
        } else if (k != null) {
            builder.field(K, k);
        } else {
            throw new IllegalArgumentException("k or minScore must be provided in the query");
        }
        if (filterName != null && filterValue != null) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);
            builder.field(filterName, filterValue);
            builder.endObject();
            builder.endObject();
        }
        builder.endObject().endObject().endObject().endObject().endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }
}
