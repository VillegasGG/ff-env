# Usar la imagen base de Ubuntu 22.04
FROM ubuntu:22.04

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*  

# Establecer python3.10 como predeterminado
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copiar el proyecto (incluyendo requirements.txt)
COPY . /usr/src/app
WORKDIR /usr/src/app

# Instalar dependencias de Python desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Soporte para renderizado gráfico (ej: gymnasium.render)
RUN apt-get update && apt-get install -y \
    python3-opengl \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]