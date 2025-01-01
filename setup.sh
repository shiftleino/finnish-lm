#!/bin/bash

docker run \
  -v ~/dev/finnish-lm/:/app/src \
  -it \
  --rm \
  -d \
  --name finnish-lm \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 64G \
  finnish-lm:latest
