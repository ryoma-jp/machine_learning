FROM debian:bookworm
RUN apt update && apt install -y gnupg
RUN echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

RUN apt update && apt -y upgrade
RUN apt update && apt install -y \
    sudo \
    wget \
    git \
    cmake \
    fonts-liberation \
    python3-pip \
    python3-tk \
    python3-picamera2 \
    python3-pil python3-pil.imagetk \
    libgl1-mesa-glx \
    unzip \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone https://github.com/hailo-ai/hailort.git && \
    cd hailort && \
    git checkout 01e4c7f && \
    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release && sudo cmake --build build --config release --target install && \
    cd hailort/libhailort/bindings/python/platform && \
    python3 setup.py bdist_wheel && \
    pip install --break-system-packages ./dist/hailort-4.18.0-cp311-cp311-linux_aarch64.whl

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} -G video ${USER_NAME}

USER ${USER_NAME}

ENTRYPOINT ["/bin/bash"]
