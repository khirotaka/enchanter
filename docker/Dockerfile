FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

WORKDIR /enchanter-build
COPY enchanter/ /enchanter-build/enchanter/
COPY README.md /enchanter-build/
COPY setup.py /enchanter-build/
COPY requirements.txt /enchanter-build/
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN pip install .

WORKDIR /workspace
