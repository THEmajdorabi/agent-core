# Minimal image for run_python. Deliberately has no third-party packages: the
# sandbox is for arithmetic, parsing and small algorithms, not for pulling in a
# dependency the model happens to know.
FROM python:3.12-slim

# nobody:nogroup (65534) already exists in the base image. The container is
# started with --user 65534:65534, so nothing here runs as root.
RUN mkdir -p /workspace && chown 65534:65534 /workspace

WORKDIR /workspace
USER 65534:65534

# Unbuffered so stdout survives a kill, and no .pyc writes to a read-only root.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-c", "print('sandbox ready')"]
