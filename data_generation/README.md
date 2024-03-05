Use this folder for data generation scripts. 

trajectory_generation_w_constraints.py: generates a trajectory with constraints. The current constraint is a rotational pose constraint.

You may open multiple dockers to run multiple instances of the script. To do so, first create a docker network:
```
docker network create --subnet=172.18.0.0/16 datagen 
```
You may replace the ip address with any other ip address. 
You may then want to create a docker image that has all the appropriate files compiled and ready. To do so, compile both catkin_ws and ws_moveit in the docker container. You may also want to install some form of terminal multiplexer in the container, such as tmux. Then, commit the docker container to an image by running the following command:
```
docker commit <container_id> constraindatagen-ubuntu18
```
You can find the docker container id by running the following command:
```
docker container ls
```
NOTE: commiting the docker container to an image will take a while and will take up a lot of space. Make sure you have enough space on your computer.


Then, run the docker with the following command:
```
docker run -v $PWD/../:/root/catkin_ws/src/jiaming_manipulation \
	-e DISPLAY=":1" \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--ipc=host \
	--gpus all \
	--network=datagen \
	-p 8880:8880 \
	--privileged=true \
	-v /etc/localtime:/etc/localtime:ro \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -p 19990:19990 -it constraindatagen-ubuntu18 bash
```
This will start one instance of the docker. You may want to start multiple instances of the docker to run multiple instances of the script. To do so, change the exposed port numbers (8880 and 19990) to different numbers and run the command again.

To run the script, open two terminals in the docker and source catkin_ws in both terminals. Then, in one terminal, run the following command:
```
roslaunch fet_moveit_config data_generation_with_moveit.launch
```
Wait for all ros services to start. Then, in the other terminal, run the following command:
```
python trajectory_generation_w_constraints.py $START_SCENE_NUMBER $NUMBER_OF_SCENES_TO_GENERATE 
```
In each running container, replace $START_SCENE_NUMBER with the starting scene number and $NUMBER_OF_SCENES_TO_GENERATE with the number of scenes to generate. For example, if you want to generate 1000 scenes, you can run the following command in one container:
```
python trajectory_generation_w_constraints.py 0 500
```
and the following command in another container:
```
python trajectory_generation_w_constraints.py 500 500
```
The scipt is able to recover progress from previous runs. For example, if you have already generated 200 scenes, running the previous command will generate 300 more scenes, starting from scene 200.
```

NOTE: RVIZ could cause memory issues when running datagen scripts. If you encounter memory issues, try running the script without RVIZ.
ANOTHER NOTE: Check your memory usage when running the script. If you run out of memory, your computer will freeze and you will have to restart your computer XD. You can try running the script with fewer scenes to test out the memory usage.