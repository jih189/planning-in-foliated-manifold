#!/bin/bash
set -ex
docker run -v $PWD/../:/root/catkin_ws/src \
	-e DISPLAY=":0" \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--ipc=host \
	--gpus all \
	--network="host" \
	-p 8888:8888 \
	-p 6006:6006 \
	--privileged=true \
	-v /etc/localtime:/etc/localtime:ro \
	-v /dev/video0:/dev/video0 \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -p 19997:19997 -it planning-in-foliation bash
