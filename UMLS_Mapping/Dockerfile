FROM python:3.10-bullseye

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Keep pip tooling modern
RUN python -m pip install --upgrade pip setuptools wheel

# Pin numpy early to avoid binary mismatches
RUN pip install --no-cache-dir numpy==1.24.4

# Set workdir before copying files (cache-friendly)
WORKDIR /app

# Copy a small docker-specific requirements file if you created one.
# If you didn't create it, this line won't hurt (build will fail if file missing — remove if you prefer).
COPY requirements.docker.txt /app/requirements.docker.txt

# Install any extra pip deps listed in requirements.docker.txt (if present)
RUN if [ -f /app/requirements.docker.txt ]; then \
      pip install --no-cache-dir -r /app/requirements.docker.txt ; \
    fi

# Install the project-specific Python dependencies (exact set you used before)
RUN pip install --no-cache-dir \
    "scikit-learn==1.1.2" \
    "spacy>=3.7.2,<4.0.0" \
    "scispacy>=0.5.3,<0.6" \
    nmslib \
    negspacy \
    pandas \
    joblib

# Install scispaCy model (BC5CDR)
RUN pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz

# Defaults (can override at runtime with -e)
ENV PREFER_TYPES="T047,T046" \
    SCORE_THRESHOLD="0.1" \
    SPACY_MODEL="en_ner_bc5cdr_md" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Copy the rest of your app code
COPY . /app

# Use ENTRYPOINT so runtime args are passed to the script (avoids the --in exec error)
ENTRYPOINT ["python", "mapConditions.py"]