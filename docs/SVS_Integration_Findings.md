

##  Missing Features

### 1. Document Filtering Support 

**Impact**: Cannot restrict searches to document subsets, breaking filtered queries, nested field search, and hybrid search scenarios.

#### Code Evidence - Where HNSW Succeeds:

**File**: `faiss/faiss/IndexHNSW.h` (Lines ~45-50)
```cpp
struct SearchParametersHNSW : SearchParameters {
    size_t efSearch = 16;
    IDSelector* sel = nullptr;      // ← Enables document filtering
    IDGrouper* grp = nullptr;       // ← Enables result grouping
    bool collect_metrics = false;
};
```

**File**: `faiss/faiss/IndexHNSW.cpp` (Lines ~200-220)
```cpp
void IndexHNSW::search(
    idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels,
    const SearchParameters* params    // ← Accepts search parameters
) const {
    const SearchParametersHNSW* hnsw_params = 
        dynamic_cast<const SearchParametersHNSW*>(params);
    
    if (hnsw_params && hnsw_params->sel) {
        // Apply document filtering during search
        hnsw.searchWithSelector(x, k, distances, labels, hnsw_params->sel);
    }
}
```

#### Code Evidence - Where SVS Fails:

**File**: `faiss/faiss/IndexSVSVamana.h` (Lines ~30-40)
```cpp
//  Missing: No SearchParametersSVS struct defined
//  Missing: No IDSelector support
```

**File**: `faiss/faiss/IndexSVSVamana.cpp` (Lines ~120-130)
```cpp
void IndexSVSVamana::search(
    idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels
    //  Missing: const SearchParameters* params parameter
) const {
    // Always performs unfiltered search
    // No way to restrict to specific document IDs
}
```

#### OpenSearch Integration Impact:

**File**: `k-NN/jni/src/faiss_wrapper.cpp` (Lines ~647-680)
```cpp
jobjectArray QueryIndex_WithFilter(..., jlongArray filterIdsJ, ...) {
    // HNSW:  Can use filters
    auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
    if (hnswReader) {
        faiss::SearchParametersHNSW hnswParams;
        hnswParams.sel = idSelector.get();  // ← Applies filter
        indexReader->search(1, queryVector, k, distances, labels, &hnswParams);
    }
    
    // SVS:  Cannot use filters  
    auto svsReader = dynamic_cast<const faiss::IndexSVSVamana*>(indexReader->index);
    if (svsReader) {
        // No way to pass filterIdsJ - ignored completely!
        indexReader->search(1, queryVector, k, distances, labels, nullptr);
        //                                                            ^^^^^^^ 
        //                                                    No filter support!
    }
}
```

### 2. Range/Radial Search Support 

**Impact**: Cannot find all vectors within a distance threshold, breaking recommendation systems and similarity-based queries.

#### Code Evidence - Where HNSW Succeeds:

**File**: `faiss/faiss/Index.h` (Lines ~200-210)
```cpp
virtual void range_search(
    idx_t n, const float* x, float radius,
    RangeSearchResult* result,
    const SearchParameters* params = nullptr
) const;
```

**File**: `faiss/faiss/IndexHNSW.cpp` (Lines ~300-320)
```cpp
void IndexHNSW::range_search(
    idx_t n, const float* x, float radius,
    RangeSearchResult* result, 
    const SearchParameters* params
) const {
    // Implementation that finds ALL vectors within radius
    hnsw.range_search(x, radius, result, params);
}
```

#### Code Evidence - Where SVS Fails:

**File**: `faiss/faiss/IndexSVSVamana.cpp`
```bash
$ grep -n "range_search" faiss/faiss/IndexSVSVamana.cpp
# No results - method doesn't exist!
```

**File**: `faiss/faiss/IndexSVSVamana.h`
```bash  
$ grep -n "range_search" faiss/faiss/IndexSVSVamana.h
# No results - method not declared!
```

#### OpenSearch Integration Impact:

**File**: `k-NN/jni/src/faiss_wrapper.cpp` (Lines ~800-830)
```cpp
jobjectArray RangeSearchWithFilter(..., jfloat radiusJ, ...) {
    // HNSW: Supports range search
    auto hnswReader = dynamic_cast<const faiss::IndexHNSW*>(indexReader->index);
    if (hnswReader) {
        faiss::RangeSearchResult rangeSearchResult(1);
        indexReader->range_search(1, queryVector, radiusJ, &rangeSearchResult, &hnswParams);
        return processRangeResults(rangeSearchResult);
    }
    
    // SVS:  No range search support
    auto svsReader = dynamic_cast<const faiss::IndexSVSVamana*>(indexReader->index);  
    if (svsReader) {
        //  IndexSVSVamana::range_search() doesn't exist!
        throw std::runtime_error("SVS indexes do not support range search");
    }
}
```

### 3. Nested Field Vector Search 

**Impact**: Cannot search vectors within nested document structures, breaking e-commerce, multi-modal, and hierarchical data use cases.

#### OpenSearch Query Example:
```json
POST /products/_search
{
  "query": {
    "nested": {
      "path": "variants",
      "query": {
        "knn": {
          "variants.feature_vector": {
            "vector": [0.1, 0.2, 0.3],
            "k": 5
          }
        }
      }
    }
  }
}
```

#### Detailed Code Path Analysis:

**File**: `k-NN/src/main/java/org/opensearch/knn/index/query/KNNWeight.java` (Lines ~150-170)
```java
public Scorer scorer(LeafReaderContext context) throws IOException {
    // Create filter for nested documents
    Bits nestedDocFilter = createNestedDocumentFilter(context, nestedPath);
    
    if (nestedDocFilter != null) {
        long[] nestedFilterIds = convertBitsToDocIds(nestedDocFilter);
        // This filter gets ignored by SVS!
        return createFilteredKNNScorer(context, nestedFilterIds);
    }
}
```

**File**: OpenSearch Core - `NestedQueryBuilder.java` (Lines ~80-100)
```java
@Override
protected Query doToQuery(QueryShardContext context) {
    // 1. Create BitSet for parent documents that contain nested fields
    BitSetProducer parentFilter = createParentFilter(path);
    
    // 2. Create BitSet for valid nested child documents  
    BitSetProducer childFilter = createChildFilter(path);
    
    // 3. Build the inner query (KNN + filters)
    Query innerQuery = query.toQuery(context);
    
    // 4. Wrap in NestedQuery that handles parent-child relationships
    return new NestedQuery(parentFilter, innerQuery, childFilter, scoreMode);
}
```

**File**: Lucene Core - `NestedQuery.java` (Lines ~120-150)
```java
private class NestedWeight extends Weight {
    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
        // Get parent and child document bits
        final Bits parentBits = parentFilter.getBitSet(context);
        final Bits childBits = childFilter.getBitSet(context);
        
        // Create combined bits: valid nested docs that match the query
        final Bits acceptDocs = createNestedAcceptBits(parentBits, childBits);
        
        // Pass these filtered bits to the KNN search - THIS IS WHERE SVS FAILS
        return new NestedScorer(childScorer, parentBits, acceptDocs);
    }
}
```

**Problem**: Nested search relies on document filtering (feature #1 above). Since SVS cannot filter documents, it returns vectors from ALL documents instead of just the matching nested documents, completely breaking nested search functionality.

**Real-world Example**:
```json
// E-commerce product catalog with nested product variants
{
  "_source": {
    "title": "Electronics Bundle",
    "products": [                    // ← Nested array
      {
        "name": "Laptop",
        "category": "computers", 
        "product_vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "price": 999
      },
      {
        "name": "Mouse", 
        "category": "accessories",
        "product_vector": [0.6, 0.7, 0.8, 0.9, 1.0], 
        "price": 25
      }
    ]
  }
}

// Query: Find similar laptops under $500 in nested products
// HNSW:  Returns only matching nested products with filters applied
// SVS: Returns random similar vectors from ANY document, breaking context
```

### 4. Runtime Search Parameter Tuning 

**Impact**: Cannot adjust search quality vs. speed trade-offs at query time, limiting operational flexibility.

#### Code Evidence - HNSW Dynamic Tuning:

**File**: `faiss/faiss/IndexHNSW.h` (Lines ~45-50)
```cpp
struct SearchParametersHNSW : SearchParameters {
    size_t efSearch = 16;    // ← Can be modified per query
};
```

**File**: `k-NN/jni/src/faiss_wrapper.cpp` (Lines ~500-520)
```cpp
// HNSW allows runtime parameter adjustment
faiss::SearchParametersHNSW hnswParams;
hnswParams.efSearch = getIntegerMethodParameter(env, methodParamsJ, EF_SEARCH, hnswParams.efSearch);
// ↑ efSearch can be different for each query
```

**OpenSearch Usage**:
```json
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 2.0],
        "k": 10,
        "method_parameters": {
          "ef_search": 200    // ← Runtime adjustable per query
        }
      }
    }
  }
}
```

#### Code Evidence - SVS Fixed Parameters:

**File**: SVS integration shows no runtime parameters:
```cpp
// SVS configurations are completely static:
"SVS,Vamana64"    // ← Degree fixed at index creation time
"SVS,Vamana32"    // ← No method_parameters supported
```

**OpenSearch Limitation**:
```json
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 2.0],
        "k": 10,
        "method_parameters": {
          "search_window": 200    // ← NOT SUPPORTED by SVS!
        }
      }
    }
  }
}
```

