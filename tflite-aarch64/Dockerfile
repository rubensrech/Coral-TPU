FROM ubuntu:20.04
ARG TENSORFLOW_COMMIT=master

USER root
RUN apt-get update && \
    apt-get install -y g++-aarch64-linux-gnu make && \
    apt-get autoremove -y

RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install unzip -y

RUN git clone https://github.com/tensorflow/tensorflow.git
RUN cd tensorflow && git checkout $TENSORFLOW_COMMIT

RUN ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
