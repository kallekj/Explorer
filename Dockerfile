# Start from a core stack version
FROM jupyter/base-notebook:latest
# Install from requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]