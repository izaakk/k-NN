# GitHub Issue: Add Intel SVS Support to OpenSearch k-NN Plugin

## Issue Title
Add Intel Scalable Vector Search (SVS) support to OpenSearch k-NN plugin via Faiss integration

## Description

This issue proposes adding support for Intel's Scalable Vector Search (SVS) library to the OpenSearch k-NN plugin. SVS is a high-performance vector search library that provides efficient graph-based indexing and search capabilities with advanced compression techniques.

## Motivation

Intel SVS offers several advantages over existing vector search methods:

1. **High Performance**: Optimized for Intel hardware with SIMD instructions
2. **Memory Efficiency**: Advanced compression techniques (LVQ, LeanVec) reduce memory footprint
3. **Scalability**: Designed for large-scale vector datasets
4. **Flexibility**: Supports both flat and graph-based (Vamana) indexing
5. **Integration**: Available through Faiss as a backend engine

## Proposed Changes

### 1. Faiss Submodule Update

**File**: `.gitmodules`

Update the Faiss submodule to point to a commit that includes SVS support:
- Current commit: needs to be updated to a Faiss commit that includes SVS integration
- Target commit: `0a62dee22bc421c28655264547fbcf99482017cb` (or latest SVS-enabled commit)

**Build Changes**:
- Update CMakeLists.txt to enable SVS compilation flags
- Ensure Intel SVS dependencies are properly linked

### 2. JNI Layer Implementation

**File**: `jni/tests/faiss_wrapper_test.cpp`

Add comprehensive test coverage for SVS configurations:
- Basic SVS,Flat functionality
- SVS,Vamana graph configurations with various degrees (32, 64, 96, 128)  
- SVS compression variants (LVQ4x0, LVQ4x4, LVQ4x8, LeanVec4x4, LeanVec4x8, LeanVec8x8)
- Multi-metric support (L2, Inner Product, Cosine)
- Edge cases and error handling

**Implementation Note**: SVS integration leverages the existing Faiss JNI interface. No changes to `faiss_wrapper.cpp` were required - SVS indexes work through the standard `createIndex` and `queryIndex` functions using SVS-specific index description strings (e.g., "SVS,Flat", "SVS,Vamana64", "SVS,Vamana32,LVQ4x4").

### 3. Java Plugin Layer Implementation

**File**: `src/main/java/org/opensearch/knn/common/KNNConstants.java`

Add SVS-specific constants:
```java
public static final String METHOD_SVS_FLAT = "svsflat";
public static final String METHOD_SVS_VAMANA = "svsvamana";
public static final String SVS_PARAMETER_DEGREE = "degree";
public static final String SVS_PARAMETER_COMPRESSION = "compression";
```

**File**: `src/main/java/org/opensearch/knn/index/engine/faiss/FaissSVSFlatMethod.java` (New)
**File**: `src/main/java/org/opensearch/knn/index/engine/faiss/FaissSVSVamanaMethod.java` (New)

Create SVS method implementations:
```java
public class FaissSVSFlatMethod extends AbstractFaissMethod {
    public static final MethodComponent SVS_FLAT_COMPONENT = MethodComponent.Builder.builder("svsflat")
        .addParameter(/* SVS flat parameters */)
        .build();
        
    @Override
    public String createIndexDescription(MethodComponentContext methodComponentContext) {
        // Generate Faiss index description strings like:
        // "SVS,Flat"
        // "SVS,Flat,LVQ4x4"
    }
}

public class FaissSVSVamanaMethod extends AbstractFaissMethod {
    public static final MethodComponent SVS_VAMANA_COMPONENT = MethodComponent.Builder.builder("svsvamana")
        .addParameter(/* SVS vamana parameters */)
        .build();
        
    @Override
    public String createIndexDescription(MethodComponentContext methodComponentContext) {
        // Generate Faiss index description strings like:
        // "SVS,Vamana64" 
        // "SVS,Vamana32,LVQ4x4"
    }
}
```

**File**: `src/main/java/org/opensearch/knn/index/engine/faiss/FaissMethodResolver.java`

Update method resolution to handle SVS methods:
```java
private ResolvedMethodContext resolveSVSFlatMethod(
    KNNMethodContext knnMethodContext,
    KNNMethodConfigContext knnMethodConfigContext, 
    SpaceType spaceType
) {
    // SVS Flat-specific resolution logic
}

private ResolvedMethodContext resolveSVSVamanaMethod(
    KNNMethodContext knnMethodContext,
    KNNMethodConfigContext knnMethodConfigContext, 
    SpaceType spaceType
) {
    // SVS Vamana-specific resolution logic
}
```

**File**: `src/main/java/org/opensearch/knn/index/engine/faiss/Faiss.java`

Add SVS methods to supported methods:
```java
SUPPORTED_METHODS = Map.of(
    METHOD_HNSW, FaissHNSWMethod.getInstance(),
    METHOD_IVF, FaissIVFMethod.getInstance(),
    METHOD_SVS_FLAT, FaissSVSFlatMethod.getInstance(),
    METHOD_SVS_VAMANA, FaissSVSVamanaMethod.getInstance()
);
```

### 4. Test Implementation

**File**: `src/test/java/org/opensearch/knn/index/engine/faiss/FaissSVSFlatMethodTests.java` (New)
**File**: `src/test/java/org/opensearch/knn/index/engine/faiss/FaissSVSVamanaMethodTests.java` (New)

Comprehensive unit tests covering:
- Parameter validation for both svsflat and svsvamana methods
- Index description generation for flat and vamana variants
- Method component initialization
- Compression parameter handling
- Error handling

**File**: `src/test/java/org/opensearch/knn/index/engine/faiss/FaissMethodResolverTests.java`

Add SVS method resolution tests:
```java  
@Test
public void testResolveSVSFlatMethod() {
    // Test SVS flat method resolution logic
}

@Test
public void testResolveSVSVamanaMethod() {
    // Test SVS vamana method resolution logic
}
```

### 5. Documentation Updates

**File**: Documentation integration into official OpenSearch docs

Add comprehensive SVS documentation covering:

**Parameters Table**:

**SVS Flat Method**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression` | string | `uncompressed` | Compression type: `uncompressed`, `LVQ4x0`, `LVQ4x4`, `LVQ4x8`, `LeanVec4x4`, `LeanVec4x8`, `LeanVec8x8` |

**SVS Vamana Method**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `degree` | integer | 64 | Graph connectivity degree (32-128) |
| `compression` | string | `uncompressed` | Compression type: `uncompressed`, `LVQ4x0`, `LVQ4x4`, `LVQ4x8`, `LeanVec4x4`, `LeanVec4x8`, `LeanVec8x8` |

**Configuration Examples**:

SVS Flat (exhaustive search):
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "engine": "faiss",
    "name": "svsflat"
  }
}
```

SVS Flat with compression:
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "engine": "faiss",
    "name": "svsflat",
    "parameters": {
      "compression": "LVQ4x4"
    }
  }
}
```

SVS Vamana (graph-based):
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "engine": "faiss",
    "name": "svsvamana",
    "parameters": {
      "degree": 64
    }
  }
}
```

SVS Vamana with compression:
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "engine": "faiss",
    "name": "svsvamana",
    "parameters": {
      "degree": 64,
      "compression": "LeanVec8x8"
    }
  }
}
```

## Implementation Plan

### Phase 1: Infrastructure Setup
1. Update Faiss submodule to SVS-enabled commit
2. Verify build system supports SVS compilation
3. Update build documentation

### Phase 2: JNI Testing and Validation
1. Implement comprehensive test suite in faiss_wrapper_test.cpp  
2. Validate all SVS configurations work correctly through existing JNI interface
3. Verify SVS index creation, training, and search functionality

### Phase 3: Java Plugin Integration
1. Create FaissSVSFlatMethod and FaissSVSVamanaMethod classes with parameter handling
2. Update FaissMethodResolver for both SVS method resolutions  
3. Add both SVS methods to Faiss engine's supported methods
4. Implement comprehensive unit tests for both methods

### Phase 4: Documentation and Validation
1. Create user-facing documentation with examples
2. Add parameter reference documentation
3. Validate integration with OpenSearch cluster
4. Performance benchmarking

## Testing Strategy

### Unit Tests
- Parameter validation for all SVS configurations
- Index description generation correctness
- Method resolution logic validation
- Error handling for invalid parameters

### Integration Tests  
- JNI layer functionality with various SVS configurations
- End-to-end index creation and search operations
- Multi-metric support validation
- Compression variants testing

### Performance Tests
- Memory usage comparison with/without compression
- Search latency benchmarks
- Index creation time measurements
- Scalability testing with large datasets

## Compatibility

### Supported Configurations
- **Method Names**: svsflat, svsvamana
- **Graph Degrees** (svsvamana only): 32, 64, 96, 128 (default: 64)
- **Compression** (both methods): LVQ4x0, LVQ4x4, LVQ4x8, LeanVec4x4, LeanVec4x8, LeanVec8x8
- **Metrics**: L2 (Euclidean), Inner Product, Cosine


## Breaking Changes

No breaking changes expected. SVS support is additive and doesn't affect existing functionality.

## Future Enhancements

The following features are planned for future releases after the initial SVS implementation:

### 1. SVS IVF Method Support

**New Method**: `svsivf`
- **Description**: Adds support for SVS's IVF (Inverted File) clustering approach, providing an alternative to the flat and Vamana methods
- **Key Parameter**: `clusters` (integer) - Number of clusters for IVF partitioning (replaces `degree` parameter used in Vamana)
- **Benefits**: Better memory efficiency for large-scale datasets through clustering-based partitioning

**Example Configuration**:
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "engine": "faiss", 
    "name": "svsivf",
    "parameters": {
      "clusters": 256,
      "compression": "LeanVec8x8"
    }
  }
}
```

### 2. Enhanced LeanVec Compression Parameters

**Additional Parameter for LeanVec**: `dimensions`
- **Description**: When LeanVec compression types are selected (`LeanVec4x4`, `LeanVec4x8`, `LeanVec8x8`), users can specify custom dimension parameters for fine-tuned compression control
- **Type**: integer
- **Usage**: Provides more granular control over LeanVec compression behavior

**Example Configuration**:
```json
{
  "type": "knn_vector", 
  "dimension": 768,
  "method": {
    "engine": "faiss",
    "name": "svsvamana", 
    "parameters": {
      "degree": 64,
      "compression": "LeanVec8x8",
      "dimensions": 32
    }
  }
}
```

**Index Factory Translation**: The above configuration will be translated to the index factory string: `"SVSVamana,LeanVec8x8_32_OOD"`

### 3. Index Factory String Format Update

**Current Format**: `"SVS,Flat,LVQ4x4"`
**Future Format**: `"SVSFlat,LVQ4x4"`

**Changes**:
- Remove comma separation between SVS and method type
- Consolidate to single compound method names (e.g., `SVSFlat`, `SVSVamana`, `SVSIVF`)
- Maintain comma separation only between method and compression parameters
- Update `createIndexDescription()` methods in all SVS implementations to use new format


## Success Criteria

1. ✅ Faiss submodule successfully updated to SVS-enabled commit
2. ✅ All existing Faiss tests continue to pass  
3. ✅ Comprehensive SVS test suite implemented and passing
4. ✅ SVS methods successfully integrated into Java plugin layer
5. ✅ Complete documentation with working examples
6. ✅ Performance benchmarks show expected SVS benefits
