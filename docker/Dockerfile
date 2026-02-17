# Multi-stage Dockerfile for OpenSearch 3.4.0 with custom k-NN plugin (SVS enabled)
# Uses Ubuntu 24.04 throughout for GLIBC 2.39 compatibility
#
# Build from k-NN repo root:
#   docker build -f docker/Dockerfile -t opensearch-knn-svs:3.4.0 .

# Stage 1: Build the k-NN plugin with JNI libraries
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    zip \
    git \
    cmake \
    g++ \
    build-essential \
    gfortran \
    zlib1g-dev \
    libopenblas-dev \
    liblapack-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Java 21 via sdkman
RUN curl -s "https://get.sdkman.io" | bash
SHELL ["/bin/bash", "-c"]
RUN source ~/.sdkman/bin/sdkman-init.sh && \
    sdk install java 21.0.2-open && \
    sdk default java 21.0.2-open

ENV JAVA_HOME=/root/.sdkman/candidates/java/current
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy the k-NN source code (context is repo root)
WORKDIR /build
COPY . /build/k-NN

# Set git config
RUN git config --global user.email "builder@local" && \
    git config --global user.name "Builder"

WORKDIR /build/k-NN

# Initialize submodules
RUN git submodule update --init -- jni/external/nmslib || true
RUN git submodule update --init -- jni/external/faiss || true

# Build the plugin (Java)
RUN source ~/.sdkman/bin/sdkman-init.sh && \
    ./gradlew assemble -x test -x integTest \
    -Dbuild.snapshot=true \
    -Dbuild.lib.commit_patches=false

# Build JNI libraries with SVS enabled (Faiss only, skip NMSLIB)
# Build AVX512_SPR variant (for Sapphire Rapids and newer CPUs)
RUN source ~/.sdkman/bin/sdkman-init.sh && \
    rm -rf jni/build && \
    cmake -S jni -B jni/build \
    -DKNN_PLUGIN_VERSION=3.4.0-SNAPSHOT \
    -DAPPLY_LIB_PATCHES=false \
    -DCONFIG_FAISS=ON \
    -DCONFIG_NMSLIB=OFF \
    -DAVX512_SPR_ENABLED=true && \
    cmake --build jni/build --target opensearchknn_faiss opensearchknn_common opensearchknn_util --parallel $(nproc)

# Stage 2: Create OpenSearch image on Ubuntu 24.04 (for GLIBC compatibility)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENSEARCH_VERSION=3.4.0
ENV OPENSEARCH_HOME=/usr/share/opensearch
ENV OPENSEARCH_PATH_CONF=${OPENSEARCH_HOME}/config
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=${OPENSEARCH_HOME}/bin:${JAVA_HOME}/bin:${PATH}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    curl \
    wget \
    tar \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract OpenSearch 3.4.0
RUN cd /tmp && \
    wget https://artifacts.opensearch.org/releases/bundle/opensearch/${OPENSEARCH_VERSION}/opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz && \
    tar -xzf opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz && \
    mv opensearch-${OPENSEARCH_VERSION} ${OPENSEARCH_HOME} && \
    rm opensearch-${OPENSEARCH_VERSION}-linux-x64.tar.gz

# Create opensearch user and group
RUN groupadd -g 1001 opensearch && \
    useradd -u 1001 -g 1001 -d ${OPENSEARCH_HOME} -s /bin/bash opensearch

# Copy the entrypoint script from official image
COPY docker/opensearch-docker-entrypoint.sh ${OPENSEARCH_HOME}/
RUN chmod +x ${OPENSEARCH_HOME}/opensearch-docker-entrypoint.sh

# Remove bundled k-NN and neural-search plugins
RUN rm -rf ${OPENSEARCH_HOME}/plugins/opensearch-knn && \
    rm -rf ${OPENSEARCH_HOME}/plugins/opensearch-neural-search

# Copy the plugin zip from builder and install
COPY --from=builder /build/k-NN/build/distributions/opensearch-knn-3.4.0.0-SNAPSHOT.zip /tmp/
RUN export OPENSEARCH_JAVA_HOME=${JAVA_HOME} && \
    ${OPENSEARCH_HOME}/bin/opensearch-plugin install --batch file:///tmp/opensearch-knn-3.4.0.0-SNAPSHOT.zip && \
    rm /tmp/opensearch-knn-3.4.0.0-SNAPSHOT.zip

# Copy native libraries with SVS support
RUN mkdir -p ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib
COPY --from=builder /build/k-NN/jni/build/release/libopensearchknn_common.so ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/
COPY --from=builder /build/k-NN/jni/build/release/libopensearchknn_faiss_avx512_spr.so ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/
COPY --from=builder /build/k-NN/jni/build/release/libopensearchknn_util.so ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/

# Copy SVS runtime library and create symlinks
COPY --from=builder /build/k-NN/jni/build/_deps/svs-src/lib/libsvs_runtime.so.0.1.0 ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/
RUN cd ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib && \
    ln -s libsvs_runtime.so.0.1.0 libsvs_runtime.so.0 && \
    ln -s libsvs_runtime.so.0 libsvs_runtime.so

# Copy OpenMP library
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/

# Set library path
ENV LD_LIBRARY_PATH=${OPENSEARCH_HOME}/plugins/opensearch-knn/lib

# Verify libraries are in place
RUN ls -la ${OPENSEARCH_HOME}/plugins/opensearch-knn/lib/

# Set ownership
RUN chown -R opensearch:opensearch ${OPENSEARCH_HOME}

# Switch to opensearch user
USER opensearch
WORKDIR ${OPENSEARCH_HOME}

# Expose ports
EXPOSE 9200 9600

# Use the official entrypoint script
ENTRYPOINT ["./opensearch-docker-entrypoint.sh"]
CMD ["opensearch"]
