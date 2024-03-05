#!/bin/bash
set -ex
docker run -v $PWD/../:/root/catkin_ws/src/jiaming_manipulation \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--gpus all \
	--privileged=true \
	-v /etc/localtime:/etc/localtime:ro \
	-it jiaming-ubuntu18:evaluation bash
