#!/bin/bash
setfacl -m user:1000:r ${HOME}/.Xauthority

# Get the current display number
DISPLAY_NUM=$(echo $DISPLAY | cut -d: -f2 | cut -d. -f1)

# Allow X server connections
xhost +local:

# Check if GPU is available
GPU_FLAGS=""
SOFTWARE_RENDER=""
if [ -e /dev/dri ]; then
    GPU_FLAGS="--device=/dev/dri"
    echo "GPU detected, using hardware rendering"
else
    SOFTWARE_RENDER="-e LIBGL_ALWAYS_SOFTWARE=1"
    echo "No GPU detected, using software rendering (Mesa llvmpipe)"
fi

# Run container
docker run \
    -it \
    --rm \
    --user $(id -u):$(id -g) \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/home/user/.Xauthority \
    $GPU_FLAGS \
    $SOFTWARE_RENDER \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$HOME/.Xauthority:/home/user/.Xauthority" \
    --volume="$PWD:/SfS" \
    --volume="$PWD/Dataset:/Dataset" \
    sfs:latest

# Cleanup
xhost -local:
