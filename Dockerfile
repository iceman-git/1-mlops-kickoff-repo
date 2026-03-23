# =============================================================================
# Dockerfile — Lean serving image for the classification API.
# Only includes what's needed to run the FastAPI service.
# Build uses conda-lock.yml for a fully reproducible environment.
# =============================================================================

FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# ── Install dependencies from lock file ──────────────────────────────────────
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Activate the mlops environment for all subsequent commands
SHELL ["conda", "run", "-n", "mlops", "/bin/bash", "-c"]

# ── Copy only what the serving layer needs ────────────────────────────────────
COPY config.yaml .
COPY src/ ./src/
COPY models/ ./models/

# ── Expose port and start the API ─────────────────────────────────────────────
EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "mlops", \
     "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]