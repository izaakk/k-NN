/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Map;

/**
 * Pins the query-time rejection paths of {@link SvsNativeEngineService}. All of these throw before any
 * native call, so this suite runs without the SVS native library. (Nested queries no longer reject here:
 * parentIds route through to the native IDGrouper path, covered by the nested IT.)
 */
public class SvsNativeEngineServiceTests extends OpenSearchTestCase {

    private final SvsNativeEngineService service = new SvsNativeEngineService();

    /**
     * The SVS index only accepts a strictly positive faiss-domain radius; the subset of inner-product and
     * cosine thresholds that convert to radius &lt;= 0 must be rejected with a descriptive message rather
     * than an opaque native error.
     */
    public void testRadiusQueryIndex_whenNonPositiveRadius_thenRejected() {
        for (float radius : new float[] { 0.0f, -0.4f }) {
            UnsupportedOperationException e = expectThrows(
                UnsupportedOperationException.class,
                () -> service.radiusQueryIndex(1L, new float[] { 1f }, radius, Map.of(), 10000, null, 0, null)
            );
            assertTrue(e.getMessage(), e.getMessage().contains("non-positive radius"));
        }
    }
}
