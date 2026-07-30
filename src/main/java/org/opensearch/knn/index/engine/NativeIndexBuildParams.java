/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.annotation.ExperimentalApi;

/**
 * The build side parameters of a {@link NativeEngineService} operation. The core supplied values are
 * typed fields and a new field can be added without breaking implementors, which method parameters
 * cannot do. Not every operation reads every field, numDocs feeds initIndex, dim feeds initIndex,
 * insertToIndex and createIndexFromTemplate, skipFlat feeds writeIndex.
 */
@ExperimentalApi
public record NativeIndexBuildParams(long numDocs, int dim, boolean skipFlat, EngineParameters engineParameters) {
}
