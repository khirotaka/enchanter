FROM khirotaka0122/torch:1.8.0-aarch64-py3.8.8

SHELL ["/bin/bash", "-c"]

WORKDIR /enchanter-build
COPY enchanter/ /enchanter-build/enchanter/
COPY README.md pyproject.toml /enchanter-build/

RUN poetry install --no-interaction --no-ansi

WORKDIR /workspace
