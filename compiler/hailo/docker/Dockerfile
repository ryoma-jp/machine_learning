FROM ubuntu:22.04

# Install necessary packages for Python 3.10
RUN apt update && apt install -y software-properties-common

# Add Deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y python3.10 python3.10-distutils python3-pip python3.10-dev

# Install necessary packages
RUN apt update && apt install -y graphviz-dev git libgl1 vim cmake sudo
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install HailoRT from source
RUN cd /tmp/ && \
        git clone https://github.com/hailo-ai/hailort.git && \
        cd hailort && \
        git checkout 3d673252095c11a99147a6e72b4c11150ebd0882 && \
        cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release && sudo cmake --build build --config release --target install

        # Copy Hailo Dataflow Compiler and Hailo Model Zoo wheel files
COPY ./whl/hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl /tmp/hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl
COPY ./whl/hailo_model_zoo-2.13.0-py3-none-any.whl /tmp/hailo_model_zoo-2.13.0-py3-none-any.whl

# Install wheel files (local user)
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
USER ${USER_NAME}

ENV PATH="${PATH}:/home/${USER_NAME}/.local/bin"
RUN pip3 install /tmp/hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl /tmp/hailo_model_zoo-2.13.0-py3-none-any.whl

USER root

# Download hailo model zoo
RUN cd /tmp/ && \
        git clone https://github.com/hailo-ai/hailo_model_zoo.git && \
        cd hailo_model_zoo && \
        git checkout 96ae8c593e6874474a823c04eaca9cf1cc3a3689
