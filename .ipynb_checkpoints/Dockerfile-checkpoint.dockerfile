FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget

# Clone SAM2 directly
WORKDIR /opt/ml/code
RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR /opt/ml/code/sam2
RUN pip install -e .
RUN python setup.py build_ext --inplace

# Go back to root working directory
WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY config/sam2_config.yaml ./config/sam2_config.yaml
COPY utils ./utils

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENTRYPOINT ["python", "inference.py"]
