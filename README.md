# Planning in Foliated Manifolds

This is the project to develop a planning framework for foliated manifolds. We separate the system into two sections. One is the task planner, while another is motion planner which is using [Moveit](https://github.com/jih189/moveit_cbirrt) which we modified to work with foliated manifolds. However, this part is in the docker container, so you can search more information of the motion planner in that repo.

Components of this project are:
* <h3>task_planner</h3>
    The actual planning framework we are developing.

* <h3>docker_image</h3>
    We do provide the docker file where all the setup is ready to use.

* <h3>fetch_ros</h3>
    This is the configuration to allow the motion planner to work with the Fetch robot.

## Installation
We provide the docker file to use this planning framework by following the steps below.
```
cd docker_image
sh build.sh
xhost +
sh run.sh
```

Once entering the docker container, you can use the following command to build the project.
```
./prepare_workspace.sh
cd catkin_ws
source devel/setup.bash
```

You can use the following command to enter the docker container.
```
sh enter_lastest_container.sh
```