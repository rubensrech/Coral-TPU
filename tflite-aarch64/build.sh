TENSORFLOW_COMMIT=48c3bae94a8b324525b45f157d638dfd4e8c3be1

HOST_LIB_DIR=$(pwd)/lib
IMAGE_TAG=tflite-aarch64-builder
CONTAINER_TFLITE_MAKE_DIR="/tensorflow/tensorflow/lite/tools/make"
CONTAINER_TFLITE_LIB_DIR="$CONTAINER_TFLITE_MAKE_DIR/gen/linux_aarch64/lib"

docker build -t $IMAGE_TAG . --build-arg TENSORFLOW_COMMIT=$TENSORFLOW_COMMIT

mkdir -p lib
docker run --rm -v $HOST_LIB_DIR:$CONTAINER_TFLITE_LIB_DIR $IMAGE_TAG $CONTAINER_TFLITE_MAKE_DIR/build_aarch64_lib.sh