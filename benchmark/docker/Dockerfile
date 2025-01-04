FROM nvcr.io/nvidia/pytorch:24.07-py3

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
