# The agent runtime itself. The sandbox image it shells out to is a separate,
# much smaller image built from docker/sandbox.Dockerfile.
FROM python:3.12-slim

# docker-cli only: the container talks to the host daemon over the mounted
# socket rather than running its own. Docker-in-Docker is not needed here and
# would require --privileged.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md config.yaml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY scripts/ ./scripts/
COPY benchmark/ ./benchmark/

ENTRYPOINT ["agent-core"]
CMD ["--help"]
