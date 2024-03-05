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

# # install the coppeliasim
RUN apt-get update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        vim tar xz-utils \
        libx11-6 libxcb1 libxau6 libgl1-mesa-dev \
        xvfb dbus-x11 x11-utils libxkbcommon-x11-0 \
        libavcodec-dev libavformat-dev libswscale-dev \
	libsodium23 libopenexr22 \
	libraw1394-11 libraw1394-dev libraw1394-tools \
	libusb-1.0-0 libusb-1.0-0-dev \
        && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /shared /opt

COPY ./download/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu18_04.tar.xz /root/
RUN tar -xf /root/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu18_04.tar.xz -C /root && \
    rm /root/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu18_04.tar.xz && cd /root && mv CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu18_04 CoppeliaSim

ENV COPPELIASIM_ROOT_DIR=/root/CoppeliaSim

EXPOSE 19997

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

# setup the python client for remote control in Coppeliasim
RUN python3 -m pip install pyzmq
RUN python3 -m pip install cbor
RUN git clone --recursive https://github.com/CoppeliaRobotics/zmqRemoteApi && cd zmqRemoteApi && git fetch && git checkout tags/coppeliasim-v4.4.0-rev0 && \
    mkdir -p build && cd build && cmake -DMAKE_BUILD_TYPE=Release .. && cmake --build . && cmake --install .

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

# install libtorch
RUN apt-get install unzip && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip && unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip

# include CUDA to path
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH $CUDA_HOME/bin:$PATH

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
    apt-get install -y ros-melodic-trac-ik-lib ros-melodic-costmap-2d ros-melodic-rail-manipulation-msgs ros-melodic-joint-trajectory-controller ros-melodic-teleop-twist-keyboard ros-melodic-base-local-planner ros-melodic-costmap-converter ros-melodic-mbf-costmap-core ros-melodic-mbf-msgs ros-melodic-teb-local-planner

# Install fcl library
RUN git clone https://github.com/flexible-collision-library/fcl.git && cd fcl && mkdir build && cd build && cmake .. && make && make install

# install trimesh
RUN apt-get install -y python-scipy python-rtree

# # install blender
# RUN apt-get update && apt-get install -y libc6 libc6-dev

# COPY ./download/blender-3.4.0-linux-x64.tar.xz /root/

# RUN tar -xf /root/blender-3.4.0-linux-x64.tar.xz -C /root && \
#     rm /root/blender-3.4.0-linux-x64.tar.xz && cd /root && mv blender-3.4.0-linux-x64 Blender

# # Download the Python Source since it is not bundled with Blender
# RUN wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz \
# 	&& tar -xzf Python-3.10.8.tgz \
# 	&& cp -r Python-3.10.8/Include/* /root/Blender/3.4/python/include/python3.10/ \
# 	&& rm -rf Python-3.10.8.tgz \
# 	&& rm -rf Python-3.10.8
	
# # install ros related packages in blender
# RUN /root/Blender/3.4/python/bin/python3.10 -m ensurepip && /root/Blender/3.4/python/bin/python3.10 -m pip install pyyaml rospkg opencv-python netifaces

# # set alias for blender
# RUN echo "alias blender='$HOME/Blender/blender'" >> $HOME/.bashrc

# install conda
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
ENV PATH=$PATH:$CONDA_DIR/bin

# Create a conda env for contact graspnet
COPY ./download/contact_graspnet_env.tar.xz /root/
RUN tar -xf /root/contact_graspnet_env.tar.xz -C /root && rm /root/contact_graspnet_env.tar.xz
RUN conda env create -f contact_graspnet_env.yml

# Create a conda env for contact graspnet for 30 series
COPY ./download/contact_graspnet_30.tar.xz /root/
RUN tar -xf /root/contact_graspnet_30.tar.xz -C /root && rm /root/contact_graspnet_30.tar.xz
RUN conda env create -f contact_graspnet_30.yml

RUN echo 'source $CONDA_DIR/etc/profile.d/conda.sh' >> $HOME/.bashrc

# install python package for tensorflow
RUN apt-get install -y ros-melodic-ros-numpy 

# install python3 package required for ROS
RUN apt-get install -y python3-pip python3-yaml python3-netifaces && pip3 install rospkg catkin_pkg

# # recompile tf2_ros for python3
RUN apt-get install -y python3-catkin-pkg-modules python3-rospkg-modules python3-empy

RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash && mkdir -p $HOME/ros_python3_ws/src && cd $HOME/ros_python3_ws && wstool init && wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v melodic-devel && wstool up && rosdep install --from-paths src --ignore-src -y -r'
RUN sed -i '/from __future__ import absolute_import/aimport sys\nsys.path[:] = [x for x in sys.path if "python3" not in x] if sys.version_info[0] < 3 else sys.path' /root/ros_python3_ws/src/geometry2/tf2_ros/src/tf2_ros/__init__.py
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash && cd $HOME/ros_python3_ws && \
	catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so'

# install python3.8
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
   wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz && tar -xf Python-3.8.16.tgz && \
   cd Python-3.8.16 && ./configure --enable-shared --prefix /usr --enable-optimizations && make -j$(nproc) && make altinstall
RUN ln -s /usr/lib/python3.8/_sysconfigdata__linux_x86_64-linux-gnu.py  /usr/lib/python3.8/_sysconfigdata__x86_64-linux-gnu.py

# install boost 1.76
RUN apt-get install -y g++ python-dev libicu-dev libboost-all-dev
RUN wget -c 'http://sourceforge.net/projects/boost/files/boost/1.76.0/boost_1_76_0.tar.bz2/download' && tar xf download && rm download
RUN cd boost_1_76_0 && ./bootstrap.sh --with-python=/usr/bin/python3.8
RUN cd boost_1_76_0 && ./b2 --prefix=/root/boost_1_76_0 variant=release link=shared threading=multi runtime-link=shared toolset=gcc install
ENV LD_LIBRARY_PATH=/root/boost_1_76_0/lib:$LD_LIBRARY_PATH

# compile ompl with python3.8# # RUN sudo -H pip3 install -vU pygccxml pyplusplus numpy
RUN sudo -H python3.8 -m pip install -vU pygccxml pyplusplus numpy

## Install yaml-cpp parser
RUN git clone https://github.com/jbeder/yaml-cpp.git && cd yaml-cpp && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make && make install

# then install ompl
RUN cd ompl && mkdir -p python_build/Release && cd python_build/Release && \
    cmake ../.. -DPYTHON_EXEC=/usr/bin/python3.8  -DCASTXML=/root/castxml/bin/castxml -DBOOST_ROOT=/root/boost_1_76_0 -DBOOST_LIBRARYDIR=/root/boost_1_76_0/lib && make update_bindings && make -j$(nproc) && make install -j$(nproc)

# # uninstall yaml-cpp (not the original one)
RUN cd yaml-cpp/build && make uninstall

# install python3 package required for ROS
RUN apt-get install -y python3-pip python3-yaml python3-netifaces && python3.8 -m pip install rospkg catkin_pkg netifaces

# install open3d
RUN python3.8 -m pip install --ignore-installed scikit-learn PyYAML && python3.8 -m pip install open3d

# update skikit-image
RUN python3.8 -m pip install --upgrade scikit-image

# # install torch
RUN python3.8 -m pip install --upgrade pip && python3.8 -m pip install --upgrade pillow
RUN python3.8 -m pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3.8 -m pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-geometric -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install einops
RUN python3.8 -m pip install tensorboard

# # install mpnet_fetch
# RUN git clone https://github.com/jih189/mpnet_fetch.git

# # set the model files to the proper position
# COPY ./download/models.tar.xz $HOME
# RUN tar -xf $HOME/models.tar.xz -C $HOME && rm $HOME/models.tar.xz && mv $HOME/models $HOME/mpnet_fetch/pt_dir

# need to install opencv
RUN python3 -m pip install opencv-python open3d --ignore-installed PyYAML
RUN python3.8 -m pip install opencv-python open3d --ignore-installed PyYAML

# install Detic for object detection
RUN git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 && python3.8 -m pip install -e .
RUN git clone https://github.com/facebookresearch/Detic.git --recurse-submodules && cd Detic && python3.8 -m pip install -r requirements.txt

# install models for Detic
RUN cd Detic && mkdir models && wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# # install gpt4all(Need to install Vulkan first)
# # need to use g++-9
# RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:jonathonf/gcc && apt-get install -y gcc-9 g++-9
# RUN sudo add-apt-repository ppa:graphics-drivers/ppa && sudo apt upgrade -y && sudo apt install vulkan-utils -y
# RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add - && sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.204-bionic.list https://packages.lunarg.com/vulkan/1.3.204/lunarg-vulkan-1.3.204-bionic.list && \
#     sudo apt update && sudo apt install -y vulkan-sdk
# RUN git clone --recurse-submodules https://github.com/nomic-ai/gpt4all && cd gpt4all/gpt4all-backend/ && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++-9 -DCMAKE_C_COMPILER=gcc-9 .. && cmake --build . --parallel
# RUN cd gpt4all/gpt4all-bindings/python && python3.8 -m pip install -e .

# install dependencies for mpc_local_planner
# install control_box_rst
RUN apt-get install -y coinor-libipopt-dev
RUN git clone https://github.com/rst-tu-dortmund/control_box_rst.git && cd control_box_rst && mkdir build && cd build && cmake .. && make -j$(nproc) && make install

# # need to save the yolo door detector weight to the proper position
# COPY ./download/handle_weight.tar.xz $HOME
# RUN tar -xf $HOME/handle_weight.tar.xz -C $HOME && rm $HOME/handle_weight.tar.xz

# install packages needed in task planner
RUN apt-get install python-tk -y
RUN python -m pip install networkx scikit-learn tqdm trimesh==3.21.0 joblib
RUN python -m pip install python-fcl --upgrade

# put the script to prepare the workspace
COPY prepare_workspace.sh $HOME

RUN echo 'source $HOME/catkin_ws/devel/setup.bash' >> $HOME/.bashrc
RUN echo 'source $HOME/ros_python3_ws/devel/setup.bash' >> $HOME/.bashrc

RUN mkdir -p $HOME/catkin_ws/src/jiaming_manipulation
