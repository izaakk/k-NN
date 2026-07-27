/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.sandbox.svs;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.test.OpenSearchTestCase;

public class SvsLibraryTests extends OpenSearchTestCase {

    /**
     * The capability flags are the single source of truth for the core capability sets; pin the ones SVS
     * claims (and the ones it deliberately does not).
     */
    public void testCapabilityFlags() {
        assertTrue(SvsLibrary.INSTANCE.supportsIterativeBuild());
        assertTrue(SvsLibrary.INSTANCE.createsCustomSegmentFiles());
        assertTrue(SvsLibrary.INSTANCE.supportsFilters());
        assertTrue(SvsLibrary.INSTANCE.supportsRadialSearch());
        // Nested runs through the SVS runtime's IDGrouper (one best child per parent during graph search).
        assertTrue(SvsLibrary.INSTANCE.supportsNestedFields());
    }

    public void testExtension() {
        assertEquals(".svs", SvsLibrary.INSTANCE.getExtension());
    }

    /**
     * SVS indices store faiss metrics, so scoring and the radial-threshold conversions must match faiss
     * exactly per space type — these delegates are live code on the radial path.
     */
    public void testScoreAndRadialThresholdsDelegateToFaiss() {
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL }) {
            assertEquals(Faiss.INSTANCE.score(0.25f, spaceType), SvsLibrary.INSTANCE.score(0.25f, spaceType), 0.0f);
            assertEquals(
                Faiss.INSTANCE.distanceToRadialThreshold(0.5f, spaceType),
                SvsLibrary.INSTANCE.distanceToRadialThreshold(0.5f, spaceType)
            );
            assertEquals(
                Faiss.INSTANCE.scoreToRadialThreshold(0.75f, spaceType),
                SvsLibrary.INSTANCE.scoreToRadialThreshold(0.75f, spaceType)
            );
        }
    }
}
