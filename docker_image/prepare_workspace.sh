#!/bin/bash

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