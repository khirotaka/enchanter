FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

SHELL ["/bin/bash", "-c"]

WORKDIR /enchanter-build
COPY enchanter/ /enchanter-build/enchanter/
COPY README.md /enchanter-build/
COPY pyproject.toml /enchanter-build/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl=7.64.1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
RUN source ${HOME}/.poetry/env \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

ENV PATH "$HOME"/.poetry/bin:"$PATH"

WORKDIR /workspace
RUN chmod -R a+w .
