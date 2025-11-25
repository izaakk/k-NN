# Intel SVS Integration for OpenSearch k-NN

Intel SVS (Scalable Vector Search) is a high-performance vector search library optimized for Intel hardware architectures. This integration provides OpenSearch users with access to Intel's advanced vector indexing and search capabilities through the Faiss engine.

## Overview

Intel SVS offers several advantages:

- **Hardware Optimization**: Specifically tuned for Intel CPUs with SIMD optimizations
- **Advanced Graph Algorithms**: Vamana graph construction for efficient navigation
- **Memory-Efficient Compression**: LVQ and LeanVec compression techniques for reduced memory footprint
- **High Performance**: Optimized for both indexing and search operations

## SVS Method Configuration

The SVS method is available through the Faiss engine and supports flexible configuration of graph types and compression methods.

### Basic SVS Configuration

```json
{
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "svs",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "graph_type": "Vamana",
            "graph_degree": 32
          }
        }
      }
    }
  }
}
```

## Supported Parameters

### Common Parameters

Parameter name | Required | Default | Updatable | Description
:--- | :--- | :--- | :--- | :---
`graph_type` | No | `Vamana` | No | The graph construction algorithm. Valid values are `Flat` and `Vamana`. `Flat` provides exhaustive search, while `Vamana` builds a navigable graph for approximate search.
`graph_degree` | No | `32` | No | The degree of the Vamana graph (number of connections per node). Higher values improve recall but increase memory usage and indexing time. Valid range: 1-128.
`compression` | No | `Flat` | No | Vector compression method for memory optimization. Valid values are `Flat`, `LVQ4x0`, `LVQ4x4`, `LVQ4x8`, `LeanVec4x4`, `LeanVec4x8`, `LeanVec8x8`.

### Supported Spaces

The SVS method supports the following distance metrics:

- `l2`: Euclidean distance (L2 norm)
- `innerproduct`: Inner product (dot product)
- `cosinesimil`: Cosine similarity

### Supported Data Types

- `float`: 32-bit floating-point vectors (primary supported type)

## Graph Types

### Flat Graph

The `Flat` graph type performs exhaustive search across all vectors, providing exact results with 100% recall.

```json
"parameters": {
  "graph_type": "Flat"
}
```

**Use cases:**
- Small datasets where exact search is preferred
- Scenarios requiring 100% recall
- Baseline comparisons

### Vamana Graph

The `Vamana` graph type builds a navigable small-world graph optimized for approximate nearest neighbor search.

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 64
}
```

**Use cases:**
- Large-scale vector databases
- Applications requiring fast search with high recall
- Memory-efficient approximate search

## Compression Options

SVS provides several compression techniques to reduce memory footprint while maintaining search quality.

### Flat Compression (Default)

No compression is applied. Vectors are stored in full precision.

```json
"parameters": {
  "compression": "Flat"
}
```

### LVQ (Locally-adaptive Vector Quantization)

LVQ compression reduces memory usage by compressing vector dimensions.

#### LVQ4x0
4-bit compression with no padding:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LVQ4x0"
}
```

#### LVQ4x4
4-bit compression with 4-bit padding:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LVQ4x4"
}
```

#### LVQ4x8
4-bit compression with 8-bit padding:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LVQ4x8"
}
```

### LeanVec Compression

LeanVec applies learned transformations to vectors before compression, often providing better compression ratios.

#### LeanVec4x4
4x4 LeanVec compression:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LeanVec4x4"
}
```

#### LeanVec4x8
4x8 LeanVec compression:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LeanVec4x8"
}
```

#### LeanVec8x8
8x8 LeanVec compression:

```json
"parameters": {
  "graph_type": "Vamana",
  "graph_degree": 32,
  "compression": "LeanVec8x8"
}
```

**Note:** LeanVec compression requires a training phase where the transformation matrices are computed from the indexed data. This training happens automatically during index creation.

## Complete Configuration Examples

### High-Performance Configuration

For scenarios requiring fast search with good recall:

```json
PUT /high-perf-index
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "graph_type": "Vamana",
            "graph_degree": 64,
            "compression": "Flat"
          }
        }
      }
    }
  }
}
```

### Memory-Optimized Configuration

For scenarios where memory usage is a primary concern:

```json
PUT /memory-optimized-index
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "svs",
          "space_type": "cosinesimil",
          "engine": "faiss",
          "parameters": {
            "graph_type": "Vamana",
            "graph_degree": 32,
            "compression": "LeanVec4x4"
          }
        }
      }
    }
  }
}
```

### Exact Search Configuration

For scenarios requiring 100% recall:

```json
PUT /exact-search-index
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 256,
        "method": {
          "name": "svs",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "graph_type": "Flat"
          }
        }
      }
    }
  }
}
```

## Search Operations

SVS indices support all standard k-NN search operations:

### k-NN Search

```json
POST /my-index/_search
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, 0.3, ...],
        "k": 10
      }
    }
  }
}
```

### Radial Search

```json
POST /my-index/_search
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, 0.3, ...],
        "min_score": 0.8
      }
    }
  }
}
```

## Performance Characteristics

### Memory Usage

Compression method | Memory reduction | Search accuracy impact
:--- | :--- | :---
Flat | None (baseline) | No impact
LVQ4x0 | ~4x reduction | Minimal impact
LVQ4x4 | ~3.5x reduction | Minimal impact
LVQ4x8 | ~3x reduction | Minimal impact
LeanVec4x4 | ~4x reduction | Very minimal impact
LeanVec4x8 | ~3x reduction | Very minimal impact
LeanVec8x8 | ~2x reduction | Very minimal impact

### Indexing Performance

- **Flat graph**: Fastest indexing, linear search
- **Vamana graph**: Moderate indexing time, fast search
- **LeanVec compression**: Additional training overhead during indexing

### Search Performance

- **Flat graph**: O(n) search complexity, 100% recall
- **Vamana graph**: Sublinear search complexity, high recall with proper parameters
- **Compressed indices**: Faster search due to reduced memory bandwidth requirements

## Hardware Requirements

### Minimum Requirements

- Intel x64 architecture processor
- AVX2 instruction set support
- Sufficient RAM for index storage

### Recommended Requirements

- Intel processors with AVX-512 support
- Latest generation Intel processors (Sapphire Rapids or newer) for optimal performance
- High-bandwidth memory configuration

## Best Practices

### Graph Degree Selection

- **Low dimensions (< 100)**: graph_degree = 16-32
- **Medium dimensions (100-500)**: graph_degree = 32-64
- **High dimensions (> 500)**: graph_degree = 64-128

### Compression Selection

1. Start with `Flat` compression for baseline performance
2. Use `LVQ4x4` for balanced memory/accuracy trade-off
3. Use `LeanVec4x4` for maximum memory efficiency with minimal accuracy loss
4. Evaluate accuracy impact on your specific dataset and use case

### Memory Planning

Calculate memory requirements based on:
- Number of vectors
- Vector dimensions
- Compression method
- Graph degree (for Vamana)

Example estimation for 1M vectors, 384 dimensions, Vamana64, LeanVec4x4:
```
Memory ≈ (vectors * dimensions * 4 bytes / compression_ratio) + (vectors * graph_degree * 4 bytes)
Memory ≈ (1M * 384 * 4 / 4) + (1M * 64 * 4) ≈ 640MB
```

## Troubleshooting

### Common Issues

1. **Training failures with LeanVec**: Ensure sufficient training data (at least 1000 vectors recommended)
2. **Memory errors**: Reduce graph_degree or use more aggressive compression
3. **Poor recall**: Increase graph_degree or reduce compression level

### Performance Tuning

1. **For high recall**: Increase graph_degree, use less aggressive compression
2. **For memory efficiency**: Use LeanVec compression, reduce graph_degree
3. **For indexing speed**: Use Flat graph for small datasets, moderate graph_degree for large datasets

## Migration from Other Engines

### From Faiss HNSW

Replace `hnsw` method with `svs` method:

```json
// Before (HNSW)
"method": {
  "name": "hnsw",
  "engine": "faiss",
  "parameters": {
    "ef_construction": 256,
    "m": 32
  }
}

// After (SVS)
"method": {
  "name": "svs",
  "engine": "faiss", 
  "parameters": {
    "graph_type": "Vamana",
    "graph_degree": 32,
    "compression": "Flat"
  }
}
```

### From Lucene HNSW

```json
// Before (Lucene)
"method": {
  "name": "hnsw",
  "engine": "lucene",
  "parameters": {
    "ef_construction": 100,
    "m": 16
  }
}

// After (SVS)
"method": {
  "name": "svs", 
  "engine": "faiss",
  "parameters": {
    "graph_type": "Vamana",
    "graph_degree": 16,
    "quantizer": "Flat"
  }
}
```

## Limitations

- Only supports `float` vector data type
- Remote index building not supported
- LeanVec compression requires training data
- Limited to supported distance metrics (l2, innerproduct, cosinesimil)

## Version Compatibility

- OpenSearch version: 2.x and later (with k-NN plugin)
- Intel SVS integration: Available with Faiss integration
- Hardware: Intel x64 processors with AVX2+ instruction set

---

For more information about Intel SVS, visit the [Intel SVS GitHub repository](https://github.com/IntelLabs/ScalableVectorSearch).
