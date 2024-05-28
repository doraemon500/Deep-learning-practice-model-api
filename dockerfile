FROM --platform=linux/arm64 continuumio/miniconda3:24.3.0-0

WORKDIR /root

SHELL ["/bin/bash", "--login", "-c"]

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda init bash

ADD model_api.py .					
CMD ["conda", "run", "-n", "myenv", "python", "model_api.py"]