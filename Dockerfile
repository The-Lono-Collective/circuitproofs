FROM python:3.9-slim AS base

# 1. Install needed packages, including `git`
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git wget xz-utils g++ libgmp-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# 2. Install elan (for Lean 4)
RUN curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y

# Ensure the correct PATH is set for elan and lake
ENV PATH="/root/.elan/bin:$PATH"

# 3. Create and move to /app
WORKDIR /app

# 4. Copy entire project into /app (includes lean-toolchain which elan will use)
COPY . /app

# 5. Install the Lean toolchain specified in lean-toolchain
RUN elan toolchain install $(cat lean-toolchain) && \
    elan default $(cat lean-toolchain)

# 6. Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 7. Install Python dependencies using uv
RUN uv pip install --system --no-cache -r translator/requirements.txt
RUN uv pip install --system --no-cache -r extraction/requirements.txt
RUN uv pip install --system --no-cache flask graphviz

# 8. Update lake dependencies
RUN lake update

# 9. Get Mathlib cache (downloads pre-built .olean files)
RUN lake exe cache get

# 10. Build with Lake from project root (lakefile.lean is here, with srcDir := "lean")
RUN lake build

# 11. Expose port 5000 for Flask
EXPOSE 5000

# 12. Launch your Flask web app
CMD ["python", "/app/webapp/app.py"]
