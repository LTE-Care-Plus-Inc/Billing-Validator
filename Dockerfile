FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget tar ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"; \
    ARCH="$(uname -m)"; \
    NODE_EXPORTER_VERSION="$(wget -qO- https://api.github.com/repos/prometheus/node_exporter/releases/latest | sed -n 's/.*"tag_name":[[:space:]]*"v\([^"]*\)".*/\1/p')"; \
    test -n "$NODE_EXPORTER_VERSION"; \
    if [ "$ARCH" = "x86_64" ]; then ARCH="amd64"; fi; \
    if [ "$ARCH" = "aarch64" ]; then ARCH="arm64"; fi; \
    FILE="node_exporter-${NODE_EXPORTER_VERSION}.${OS}-${ARCH}.tar.gz"; \
    URL="https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/${FILE}"; \
    wget "$URL"; \
    tar -xvf "$FILE"; \
    mv "node_exporter-${NODE_EXPORTER_VERSION}.${OS}-${ARCH}/node_exporter" /usr/local/bin/; \
    chmod +x /usr/local/bin/node_exporter; \
    rm -rf "node_exporter-${NODE_EXPORTER_VERSION}.${OS}-${ARCH}"* "$FILE"

RUN pip install --upgrade pip \
    && pip install \
        pandas>=2.0.0 \
        numpy>=1.24.0 \
        streamlit>=1.30.0 \
        openpyxl>=3.1.0 \
        XlsxWriter>=3.1.0 \
        PyMuPDF>=1.23.0

RUN groupadd --gid 10001 lteuser \
    && useradd --uid 10001 --gid lteuser --create-home --shell /usr/sbin/nologin lteuser

COPY --chown=lteuser:lteuser . .

USER 10001:10001

EXPOSE 8501
EXPOSE 9090

CMD ["sh", "-c", "/usr/local/bin/node_exporter --web.listen-address=:9090 & exec streamlit run billing_checker.py --server.address=0.0.0.0 --server.port=8501"]
