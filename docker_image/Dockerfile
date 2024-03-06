FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

WORKDIR /root/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -y autotools-dev automake libtool python3-pip vim 

# include ros melodic
RUN apt-get update && \
   apt-get install -y --no-install-recommends \
         git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
      net-tools \
  && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
		ros-melodic-desktop-full \
		ros-melodic-image-transport \
		ros-melodic-vision-msgs \
		ros-melodic-message-filters \
         python-rosdep \
         python-rosinstall \
         python-rosinstall-generator \
         python-wstool \
   && rm -rf /var/lib/apt/lists/*

RUN echo 'source /opt/ros/melodic/setup.bash' >> $HOME/.bashrc

RUN echo 'export ROS_MASTER_URI=http://localhost:11311' >> $HOME/.bashrc
RUN echo 'export ROS_HOSTNAME=localhost' >> $HOME/.bashrc

# install cmake
RUN apt-get update && \
    apt-get install -y build-essential wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz && \
    tar -xf cmake-3.22.0.tar.gz && \
    rm cmake-3.22.0.tar.gz && \
    cd cmake-3.22.0 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

# install ompl
# first, need to instal ODE
RUN git clone https://github.com/thomasmarsh/ODE.git
RUN cd ODE && mkdir -p build && cd build && cmake .. && make && make install

# second, need to install Py++
COPY ./download/castxml.tar.xz $HOME
RUN tar -xf $HOME/castxml.tar.xz -C $HOME && rm $HOME/castxml.tar.xz
ENV PATH=/root/castxml/bin:${PATH}

RUN sudo -H pip3 install -vU pygccxml pyplusplus numpy

# then install ompl
RUN git clone https://github.com/ompl/ompl.git && cd ompl && git reset --hard fca10b4bd4840856c7a9f50d1ee2688ba77e25aa && mkdir -p build/Release && cd build/Release && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/ros/melodic ../.. -DPYTHON_EXEC=/usr/bin/python3 -DCASTXML=/root/castxml/bin/castxml && make -j$(nproc) update_bindings && make -j$(nproc) && make install

# install moveit
# setup the rosdep
RUN sudo apt -y install python-wstool python-catkin-tools clang-format-10 python-rosdep
RUN mkdir -p ws_moveit/src && cd ws_moveit/src && git clone https://github.com/jih189/moveit_cbirrt.git
RUN rosdep init && rosdep update --include-eol-distros
RUN cd ws_moveit && rosdep install -y --from-paths src --ignore-src --rosdistro melodic

# remove the ompl from ROS
RUN apt-get -y remove $(apt list --installed | awk -F/ '{print $1}' | grep ompl)

# reinstall the ompl in its build directory
RUN cd ompl/build/Release && make install

# install libtorch
RUN apt-get install unzip && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip && unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip

# include CUDA to path
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH $CUDA_HOME/bin:$PATH

# compile moveit
RUN cd ws_moveit && catkin config --extend /opt/ros/melodic --install --install-space /opt/ros/melodic --cmake-args -DCMAKE_BUILD_TYPE=Release && catkin build

# # install jupyter
RUN python3 -m pip install --upgrade pip
RUN pip3 install --ignore-installed pyzmq && pip3 install jupyter

# install python2 kernel into jupyter notebook
RUN apt install python-pip -y
RUN python2 -m pip install ipykernel
RUN python2 -m ipykernel install

# Expose the port Jupyter will run on
EXPOSE 8888

# Install necessary for building
RUN apt-get update -q && \
    apt-get install -y ros-melodic-trac-ik-lib \
                       ros-melodic-costmap-2d \
                       ros-melodic-rail-manipulation-msgs \
                       ros-melodic-joint-trajectory-controller \
                       ros-melodic-teleop-twist-keyboard \ 
                       ros-melodic-base-local-planner \
                       ros-melodic-costmap-converter \
                       ros-melodic-mbf-costmap-core \
                       ros-melodic-mbf-msgs \
                       ros-melodic-teb-local-planner \
                       python-scipy \
                       python-rtree \
                       ros-melodic-ros-numpy \
                       python3-yaml \
                       python3-netifaces \
                       python-tk


# install python3 package required for ROS
RUN pip3 install rospkg catkin_pkg

# install packages needed in task planner
RUN python -m pip install networkx scikit-learn tqdm trimesh==3.21.0 joblib
RUN python -m pip install python-fcl --upgrade

# put the script to prepare the workspace
COPY prepare_workspace.sh $HOME

RUN echo 'source $HOME/catkin_ws/devel/setup.bash' >> $HOME/.bashrc

RUN mkdir -p $HOME/catkin_ws/src/