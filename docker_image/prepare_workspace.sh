#!/bin/bash

# move weight to darknet
# mv /root/handle_weight/obj.names /root/catkin_ws/src/jiaming_manipulation/door_opener/script/darknet
# mv /root/handle_weight/yolo-obj.cfg /root/catkin_ws/src/jiaming_manipulation/door_opener/script/darknet
# mv /root/handle_weight/yolo-obj.weights /root/catkin_ws/src/jiaming_manipulation/door_opener/script/darknet

# check whether the moveit code is changed remote. If so, then recompile moveit
cd /root/ws_moveit/src/moveit_cbirrt

# Fetch remote updates
git fetch

# Check if local branch is behind the remote branch
UPSTREAM=${1:-'@{u}'}
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse "$UPSTREAM")

if [ $LOCAL = $REMOTE ]; then
    echo "Up-to-date"
else
    echo "There is update in remote repo, so update and re-compile"
    git pull
    cd /root/ws_moveit
    rm -r build devel logs
    catkin config --extend /opt/ros/melodic --install --install-space /opt/ros/melodic --cmake-args -DCMAKE_BUILD_TYPE=Release
    catkin build
fi

# compile the jiaming manipulation
cd /root/catkin_ws
catkin_make

# copy the ros plugin to CoppeliaSim
cp /root/catkin_ws/devel/lib/libsimExtRosControl.so $COPPELIASIM_ROOT_DIR
cp /root/catkin_ws/devel/lib/libsimExtRosServices.so $COPPELIASIM_ROOT_DIR

# compile pointnet in tensorflow
gpuname=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
echo "==== used gpu: $gpuname ===="
case $gpuname in
        *"RTX 30"*)
	cd /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/contact_graspnet
	conda run -n contact_graspnet_30 sh compile_pointnet_tfops.sh
	#cd /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/CoM_prediction
	#conda run -n contact_graspnet_30 sh compile_pointnet_tfops.sh
        ;;
        *"RTX 20"*)
	cd /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/contact_graspnet
	conda run -n contact_graspnet_env sh compile_pointnet_tfops.sh
	#cd /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/CoM_prediction
	#conda run -n contact_graspnet_env sh compile_pointnet_tfops.sh
        ;;
        *)
        echo "unknown gpu, so no way to compile in conda"
        ;;
esac

# need to place the contact grasp net weight into the checkpoint dir
mkdir /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/contact_graspnet/checkpoints
cp /root/catkin_ws/src/jiaming_manipulation/docker_image/download/scene_test_2048_bs3_hor_sigma_001.tar.xz /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/contact_graspnet/checkpoints
cd /root/catkin_ws/src/jiaming_manipulation/ros_tensorflow/src/contact_graspnet/checkpoints
tar -xf scene_test_2048_bs3_hor_sigma_001.tar.xz
rm scene_test_2048_bs3_hor_sigma_001.tar.xz