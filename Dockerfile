# ---- Base Stage ----
FROM python:3.10.12-slim AS base

# Set the working directory in the container
WORKDIR /leif_app/
ENV DAGSTER_HOME=/leif_app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (to leverage Docker caching)
COPY ./shared_storage/python_requirements.txt /leif_app/shared_storage/python_requirements.txt

# Install dependencies using a cached pip directory
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /leif_app/shared_storage/python_requirements.txt 

# ---- Final Stage ----
FROM base AS final

# Set the working directory
WORKDIR /leif_app/

# Copy only application source code (separately from dependencies to avoid reinstalling them)
COPY ./ai4ef_train_app /leif_app/ai4ef_train_app
COPY ./ai4ef_model_app /leif_app/ai4ef_model_app/
COPY ./shared_storage/ /leif_app/shared_storage/