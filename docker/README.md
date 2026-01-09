# Docker Build for OpenSearch k-NN with SVS

Build OpenSearch 3.4.0 with custom k-NN plugin including SVS (Scalable Vector Search) support.

## Why Ubuntu 24.04?

Uses Ubuntu 24.04 as base (not official OpenSearch image) because:
- SVS requires GLIBC 2.38+ for AVX-512 optimizations
- Ubuntu 24.04 provides GLIBC 2.39
- Official OpenSearch image uses older GLIBC (~2.31)

## Build

From the k-NN repo root:

```bash
docker build -f docker/Dockerfile -t opensearch-knn-svs:3.4.0 .
```

## Run

```bash
docker run -d \
  --name opensearch-knn-svs \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "network.host=0.0.0.0" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!" \
  opensearch-knn-svs:3.4.0
```

## Test

```bash
# Cluster health
curl -k -u admin:YourStrongPassword123! https://localhost:9200/_cluster/health?pretty

# k-NN stats
curl -k -u admin:YourStrongPassword123! https://localhost:9200/_plugins/_knn/stats?pretty
```

## Push to ECR

```bash
# Login
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 637423196053.dkr.ecr.us-west-2.amazonaws.com

# Tag and push
docker tag opensearch-knn-svs:3.4.0 637423196053.dkr.ecr.us-west-2.amazonaws.com/opensearch-knn-svs:3.4.0
docker push 637423196053.dkr.ecr.us-west-2.amazonaws.com/opensearch-knn-svs:3.4.0
```

## Native Libraries Included

- `libopensearchknn_faiss_avx512_spr.so` - Faiss JNI (AVX-512 SPR)
- `libopensearchknn_common.so` - Common library
- `libopensearchknn_util.so` - Utility library
- `libsvs_runtime.so` - Intel SVS runtime
- `libgomp.so` - OpenMP
