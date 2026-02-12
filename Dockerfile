
FROM ghcr.io/prefix-dev/pixi:0.63.2-jammy

WORKDIR /workspaces/crisp_gym


# 1️⃣ Pixi metadata (drives dependency resolution)
COPY pixi.toml pixi.lock* ./

# RUN pixi install -e humble

# 2️⃣ Python project (required for editable install)
# COPY pyproject.toml ./

# 3️⃣ Activation + helper scripts
# COPY scripts ./scripts
# RUN chmod +x scripts/*.sh

# 4️⃣ Install dependencies
# RUN pixi install -e humble

# 5️⃣ Everything else (runtime content, docs, assets, examples)
# COPY examples ./examples
# COPY media ./media
# COPY README.md CHANGELOG.md LICENSE.md CLAUDE.md ./
# COPY docker-compose.yml ./
# (Dockerfile itself does not need to be copied)

# 6️⃣ Entrypoint
# ENTRYPOINT ["bash", "-c", "pixi shell-hook -e humble > /entrypoint.sh && chmod +x /entrypoint.sh && exec bash /entrypoint.sh"]
# CMD ["/bin/bash"]

# Dockerfile snippet: safe interactive entrypoint
ENTRYPOINT ["/bin/bash", "-c", "exec bash"]
CMD []

