# SVS (Scalable Vector Search) Index Guide for OpenSearch k-NN

This guide explains how to create, configure, and search SVS indexes in OpenSearch using the k-NN plugin.

## Overview

SVS provides high-performance approximate nearest neighbor (ANN) search through the Vamana graph algorithm, integrated into the FAISS engine. Compared to HNSW, SVS Vamana offers:

- Higher query throughput (up to 2x+ at equivalent recall)
- Lower search latency
- Multiple compression options (FP16, SQ8, LVQ, LeanVec)
- Configurable construction and search parameters

## Methods

### svs_flat

Exhaustive (brute-force) search. Use for small datasets or as a baseline.

```json
PUT /my-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs_flat",
          "space_type": "l2",
          "engine": "faiss"
        }
      }
    }
  }
}
```

### svs_vamana

Graph-based approximate search using the Vamana algorithm. Recommended for production use.

```json
PUT /my-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "construction_window_size": 128
          }
        }
      }
    }
  }
}
```

## Supported Space Types

| Space Type | Description |
|---|---|
| `l2` | Euclidean (L2) distance |
| `innerproduct` | Inner product (dot product) similarity |

## Construction Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `degree` | 64 | 1-256 | Maximum out-degree of the Vamana graph. Higher values improve recall at the cost of memory and build time. |
| `construction_window_size` | 128 | > 0 | Search window used during graph construction. Larger values improve graph quality but increase build time. |

## Encoders

Encoders compress stored vectors to reduce memory usage and improve search throughput.

### Flat (default)

No compression. Full float32 precision.

```json
"parameters": {
  "encoder": {
    "name": "flat"
  }
}
```

### FP16

Reduces each float32 to float16. 2x compression.

```json
"parameters": {
  "encoder": {
    "name": "svs_fp16"
  }
}
```

### SQ8

Scalar quantization to 8 bits per dimension. 4x compression.

```json
"parameters": {
  "encoder": {
    "name": "svs_sq8"
  }
}
```

### LVQ (Locally-adaptive Vector Quantization)

Configurable quantization with primary and residual codebooks. 8x compression with default settings.

```json
"parameters": {
  "encoder": {
    "name": "lvq",
    "parameters": {
      "primary_bits": 4,
      "residual_bits": 4
    }
  }
}
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `primary_bits` | 4 | 1-8 | Bits for primary quantization |
| `residual_bits` | 4 | 0-8 | Bits for residual quantization |

### LeanVec

Learned linear projections that reduce vector dimensionality before quantization. Provides adaptive compression and often the best throughput/recall tradeoff.

LeanVec requires training on representative data. Training is handled automatically through **deferred per-shard training**: the system accumulates vectors during ingestion and trains the model once enough vectors are available.

```json
"parameters": {
  "encoder": {
    "name": "leanvec",
    "parameters": {
      "primary_bits": 4,
      "residual_bits": 4,
      "dimensions": 192,
      "training_threshold": 100000,
      "initial_training_threshold": 10000
    }
  }
}
```

| Parameter | Default | Range | Description |
|---|---|---|---|
| `primary_bits` | 4 | 1-8 | Bits for primary quantization |
| `residual_bits` | 4 | 0-8 | Bits for residual quantization |
| `dimensions` | 0 (auto) | >= 0 | Target reduced dimensionality. 0 uses all dimensions. Set to ~25-50% of original dimension for best results (e.g., 192 for 768D vectors). |
| `training_threshold` | 100000 | >= 1000 or 0 | Number of vectors required for FINAL model training. 0 uses the default. |
| `initial_training_threshold` | 10000 | >= 1000 or 0 | Number of vectors for INITIAL (rough) model training. Provides early LeanVec benefits while more data accumulates. 0 uses the default. |

#### Two-threshold training

LeanVec uses a two-phase training approach:

1. **LVQ fallback**: Until `initial_training_threshold` vectors are ingested, segments are built with LVQ encoding (no training needed).
2. **INITIAL model**: Once enough vectors accumulate, an initial model is trained during a segment merge. New segments use this model immediately.
3. **FINAL model**: When a merge contains at least `training_threshold` vectors, the final high-quality model is trained. All subsequent segments use the final model.

The transition is: **LVQ** -> **INITIAL LeanVec** -> **FINAL LeanVec**

Models persist across node restarts via segment files (`.knnlvm`).

## Search Parameters

Pass these at query time to tune the recall/latency tradeoff:

```json
GET /my-index/_search
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, ...],
        "k": 10,
        "method_parameters": {
          "search_window_size": 100,
          "search_buffer_capacity": 100
        }
      }
    }
  }
}
```

| Parameter | Default | Description |
|---|---|---|
| `search_window_size` | 10 | Controls the search beam width. Larger values improve recall at the cost of latency. Start with values around 50-200 and tune based on your recall target. |
| `search_buffer_capacity` | 10 | Size of the internal result buffer. Generally set equal to `search_window_size`. |

## Complete Examples

### Basic Vamana with FP16

```json
PUT /vectors-fp16
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 2,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "construction_window_size": 128,
            "encoder": {
              "name": "svs_fp16"
            }
          }
        }
      }
    }
  }
}
```

### LeanVec with Custom Dimensions

Best for large-scale deployments where training data is plentiful. This configuration reduces 768D vectors to 192D with 4-bit quantization.

```json
PUT /vectors-leanvec
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 2,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "construction_window_size": 128,
            "encoder": {
              "name": "leanvec",
              "parameters": {
                "primary_bits": 4,
                "residual_bits": 4,
                "dimensions": 192,
                "training_threshold": 100000,
                "initial_training_threshold": 10000
              }
            }
          }
        }
      }
    }
  }
}
```

### LVQ for Low-Latency Use Cases

LVQ provides good compression without requiring training.

```json
PUT /vectors-lvq
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "encoder": {
              "name": "lvq",
              "parameters": {
                "primary_bits": 4,
                "residual_bits": 4
              }
            }
          }
        }
      }
    }
  }
}
```

## Tuning Recommendations

### Recall vs. Throughput

- **Higher recall**: Increase `search_window_size` (e.g., 100-200). Costs more latency per query.
- **Higher throughput**: Decrease `search_window_size` (e.g., 20-50). Sacrifices some recall.
- **Rule of thumb**: `search_window_size` of ~1.5x your target `k` is a good starting point. For 95%+ recall, values of 100+ are typical.

### Encoder Selection

| Encoder | Compression | Training Required | Best For |
|---|---|---|---|
| Flat | 1x | No | Small datasets, maximum accuracy |
| FP16 | 2x | No | General purpose, minimal accuracy loss |
| SQ8 | 4x | No | Memory-constrained deployments |
| LVQ | ~8x | No | Good balance of compression and accuracy |
| LeanVec | Adaptive | Yes (automatic) | Large datasets, best throughput/recall tradeoff |

### Graph Construction

- `degree`: 64 is a good default. Increase to 96 or 128 for higher recall at the cost of memory.
- `construction_window_size`: 128 is a good default. Increase for better graph quality during construction.

## Monitoring

Training statistics are available via the k-NN stats API:

```
GET /_plugins/_knn/stats
```

Relevant counters include:
- `deferred_training_requests` - Total training attempts
- `deferred_training_success` - Successful trainings
- `deferred_training_errors` - Failed trainings
- Per-quality counters: `deferred_training_initial_*`, `deferred_training_final_*`
