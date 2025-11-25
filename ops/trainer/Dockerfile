FROM python:3.11-slim

# keep python output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build deps often needed for compiled python packages
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         gcc \
         g++ \
         libgomp1 \
         libffi-dev \
         libssl-dev \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# copy requirements so layer can be cached
COPY requirements.txt /workspace/requirements.txt

# upgrade pip and install requirements (run as root so system site-packages used)
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# create an unprivileged user to avoid creating root-owned files on the host
# default UID/GID are 1000 but can be overridden at build time with --build-arg
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup || true \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser || true

# switch to non-root user for better safety
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# default workdir is /workspace (mounted by compose)

CMD ["/bin/bash"]
