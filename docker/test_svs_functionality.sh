#!/bin/bash
#
# SVS Functional Validation Test Script
# Tests all SVS methods and encoders on a local Docker instance
#

set -e

OPENSEARCH_URL="https://localhost:9200"
ADMIN_USER="admin"
ADMIN_PASS="YourStrongPassword123!"
CURL_OPTS="-k -s -u ${ADMIN_USER}:${ADMIN_PASS}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

log_test() {
    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "\n${YELLOW}[TEST ${TEST_COUNT}] $1${NC}"
}

log_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

log_fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "${RED}✗ FAIL${NC}: $1"
    if [ -n "$2" ]; then
        echo "  Response: $2"
    fi
}

wait_for_cluster() {
    echo "Waiting for OpenSearch cluster to be ready..."
    for i in {1..60}; do
        if curl $CURL_OPTS "${OPENSEARCH_URL}/_cluster/health" >/dev/null 2>&1; then
            echo "Cluster is ready!"
            sleep 5  # Give it a few more seconds to stabilize
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Cluster did not become ready within timeout"
    exit 1
}

create_index() {
    local index_name=$1
    local mapping=$2

    curl $CURL_OPTS -X DELETE "${OPENSEARCH_URL}/${index_name}" >/dev/null 2>&1 || true
    response=$(curl $CURL_OPTS -X PUT "${OPENSEARCH_URL}/${index_name}" \
        -H 'Content-Type: application/json' \
        -d "${mapping}")

    if echo "$response" | grep -q '"acknowledged":true'; then
        return 0
    else
        return 1
    fi
}

index_vectors() {
    local index_name=$1
    local field_name=$2
    local num_vectors=$3

    bulk_data=""
    for i in $(seq 1 $num_vectors); do
        # Generate random 3D vectors
        v1=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()*10}')
        v2=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()*10}')
        v3=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()*10}')

        bulk_data="${bulk_data}{\"index\":{\"_id\":\"${i}\"}}\n"
        bulk_data="${bulk_data}{\"${field_name}\":[${v1},${v2},${v3}]}\n"
    done

    echo -e "$bulk_data" | curl $CURL_OPTS -X POST "${OPENSEARCH_URL}/${index_name}/_bulk" \
        -H 'Content-Type: application/x-ndjson' \
        --data-binary @- > /dev/null

    # Refresh index
    curl $CURL_OPTS -X POST "${OPENSEARCH_URL}/${index_name}/_refresh" > /dev/null

    # Verify doc count
    doc_count=$(curl $CURL_OPTS "${OPENSEARCH_URL}/${index_name}/_count" | grep -o '"count":[0-9]*' | cut -d: -f2)
    if [ "$doc_count" -eq "$num_vectors" ]; then
        return 0
    else
        return 1
    fi
}

search_index() {
    local index_name=$1
    local field_name=$2
    local k=$3

    query="{\"query\":{\"knn\":{\"${field_name}\":{\"vector\":[1.0,1.0,1.0],\"k\":${k}}}}}"
    response=$(curl $CURL_OPTS -X GET "${OPENSEARCH_URL}/${index_name}/_search" \
        -H 'Content-Type: application/json' \
        -d "${query}")

    hit_count=$(echo "$response" | grep -o '"total":{"value":[0-9]*' | cut -d: -f3 | head -1)
    if [ -n "$hit_count" ] && [ "$hit_count" -ge "$k" ]; then
        return 0
    else
        return 1
    fi
}

cleanup_index() {
    local index_name=$1
    curl $CURL_OPTS -X DELETE "${OPENSEARCH_URL}/${index_name}" >/dev/null 2>&1 || true
}

# ==============================================================================
# Test 1: Cluster Health and k-NN Stats
# ==============================================================================

log_test "Cluster Health Check"
health=$(curl $CURL_OPTS "${OPENSEARCH_URL}/_cluster/health")
if echo "$health" | grep -q '"status":"green"\|"status":"yellow"'; then
    log_pass "Cluster is healthy"
else
    log_fail "Cluster is not healthy" "$health"
fi

log_test "k-NN Plugin Stats"
stats=$(curl $CURL_OPTS "${OPENSEARCH_URL}/_plugins/_knn/stats")
if echo "$stats" | grep -q '"script_compilations"'; then
    log_pass "k-NN plugin is loaded"
else
    log_fail "k-NN plugin is not responding" "$stats"
fi

# ==============================================================================
# Test 2: SVS Flat Method (Basic)
# ==============================================================================

log_test "SVS Flat - Basic Index Creation and Search"
index_name="test-svs-flat"
field_name="test_vector"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
        "method": {
          "name": "svs_flat",
          "space_type": "l2",
          "engine": "faiss"
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 10; then
        log_pass "Vectors indexed successfully (10 docs)"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 3: SVS Vamana Method (Basic, no encoder)
# ==============================================================================

log_test "SVS Vamana - Basic Index with default (Flat) encoder"
index_name="test-svs-vamana-basic"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
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
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 20; then
        log_pass "Vectors indexed successfully (20 docs)"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 4: SVS Vamana + FP16 Encoder
# ==============================================================================

log_test "SVS Vamana + FP16 Encoder"
index_name="test-svs-vamana-fp16"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "encoder": {
              "name": "svs_fp16"
            }
          }
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 20; then
        log_pass "Vectors indexed successfully"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 5: SVS Vamana + SQ8 Encoder
# ==============================================================================

log_test "SVS Vamana + SQ8 Encoder"
index_name="test-svs-vamana-sq8"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "encoder": {
              "name": "svs_sq8"
            }
          }
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 20; then
        log_pass "Vectors indexed successfully"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 6: SVS Vamana + LVQ Encoder
# ==============================================================================

log_test "SVS Vamana + LVQ Encoder (4x4 bits)"
index_name="test-svs-vamana-lvq"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
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
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 20; then
        log_pass "Vectors indexed successfully"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 7: SVS Vamana + LeanVec Encoder (Direct Creation)
# ==============================================================================

log_test "SVS Vamana + LeanVec Encoder (Direct Creation with low thresholds)"
index_name="test-svs-vamana-leanvec"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {
            "degree": 64,
            "encoder": {
              "name": "leanvec",
              "parameters": {
                "primary_bits": 4,
                "residual_bits": 4,
                "dimensions": 64,
                "initial_training_threshold": 1000,
                "training_threshold": 5000
              }
            }
          }
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    # For LeanVec, index more vectors (but still small for testing)
    # With low thresholds, should use LVQ fallback for small datasets
    bulk_data=""
    for i in $(seq 1 50); do
        # Generate random 128D vectors
        vector="["
        for d in $(seq 1 128); do
            val=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()}')
            vector="${vector}${val}"
            [ $d -lt 128 ] && vector="${vector},"
        done
        vector="${vector}]"

        bulk_data="${bulk_data}{\"index\":{\"_id\":\"${i}\"}}\n"
        bulk_data="${bulk_data}{\"test_vector\":${vector}}\n"
    done

    echo -e "$bulk_data" | curl $CURL_OPTS -X POST "${OPENSEARCH_URL}/${index_name}/_bulk" \
        -H 'Content-Type: application/x-ndjson' \
        --data-binary @- > /dev/null

    curl $CURL_OPTS -X POST "${OPENSEARCH_URL}/${index_name}/_refresh" > /dev/null

    doc_count=$(curl $CURL_OPTS "${OPENSEARCH_URL}/${index_name}/_count" | grep -o '"count":[0-9]*' | cut -d: -f2)
    if [ "$doc_count" -eq "50" ]; then
        log_pass "Vectors indexed successfully (50 docs, 128D)"

        # Search with 128D query vector
        query_vector="["
        for d in $(seq 1 128); do
            query_vector="${query_vector}0.5"
            [ $d -lt 128 ] && query_vector="${query_vector},"
        done
        query_vector="${query_vector}]"

        query="{\"query\":{\"knn\":{\"test_vector\":{\"vector\":${query_vector},\"k\":5}}}}"
        response=$(curl $CURL_OPTS -X GET "${OPENSEARCH_URL}/${index_name}/_search" \
            -H 'Content-Type: application/json' \
            -d "${query}")

        hit_count=$(echo "$response" | grep -o '"total":{"value":[0-9]*' | cut -d: -f3 | head -1)
        if [ -n "$hit_count" ] && [ "$hit_count" -ge "5" ]; then
            log_pass "Search returned results (LVQ fallback used for small dataset)"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors (expected 50, got $doc_count)"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 8: Search with Method Parameters
# ==============================================================================

log_test "SVS Vamana - Search with custom search_window_size"
index_name="test-svs-vamana-params"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
        "method": {
          "name": "svs_vamana",
          "space_type": "l2",
          "engine": "faiss",
          "parameters": {"degree": 64}
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully"

    if index_vectors "$index_name" "$field_name" 30; then
        log_pass "Vectors indexed successfully"

        # Search with custom search parameters
        query='{
          "query": {
            "knn": {
              "test_vector": {
                "vector": [1.0, 1.0, 1.0],
                "k": 5,
                "method_parameters": {
                  "search_window_size": 50,
                  "search_buffer_capacity": 50
                }
              }
            }
          }
        }'

        response=$(curl $CURL_OPTS -X GET "${OPENSEARCH_URL}/${index_name}/_search" \
            -H 'Content-Type: application/json' \
            -d "${query}")

        hit_count=$(echo "$response" | grep -o '"total":{"value":[0-9]*' | cut -d: -f3 | head -1)
        if [ -n "$hit_count" ] && [ "$hit_count" -ge "5" ]; then
            log_pass "Search with method_parameters returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Test 9: Inner Product Space Type
# ==============================================================================

log_test "SVS Vamana with Inner Product space type"
index_name="test-svs-vamana-ip"
mapping='{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "test_vector": {
        "type": "knn_vector",
        "dimension": 3,
        "method": {
          "name": "svs_vamana",
          "space_type": "innerproduct",
          "engine": "faiss",
          "parameters": {"degree": 64}
        }
      }
    }
  }
}'

if create_index "$index_name" "$mapping"; then
    log_pass "Index created successfully with innerproduct space"

    if index_vectors "$index_name" "$field_name" 20; then
        log_pass "Vectors indexed successfully"

        if search_index "$index_name" "$field_name" 5; then
            log_pass "Search returned results"
        else
            log_fail "Search did not return expected results"
        fi
    else
        log_fail "Failed to index vectors"
    fi
else
    log_fail "Failed to create index"
fi
cleanup_index "$index_name"

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "=========================================="
echo "SVS Functional Validation Summary"
echo "=========================================="
echo -e "Total Tests:  ${TEST_COUNT}"
echo -e "${GREEN}Passed:       ${PASS_COUNT}${NC}"
echo -e "${RED}Failed:       ${FAIL_COUNT}${NC}"
echo "=========================================="

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. See output above for details.${NC}"
    exit 1
fi
