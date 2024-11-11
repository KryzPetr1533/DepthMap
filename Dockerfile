FROM ros:humble

# cool terminal
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# workdir
WORKDIR /rosslam

# editor + camera tools (v4l) + cuda (build)
RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    nano \
    v4l-utils \
    build-essential \
    python3-colcon-clean \
    wget \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-rendering \
    ros-humble-rviz-visual-tools \
    ros-humble-rviz-default-plugins \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    python3.10-venv \
    python3-pip && python3 -m pip install --upgrade pip && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-6 && \
    rm -rf /var/lib/apt/lists/*
    
# Set environment variables for CUDA
ENV PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# Copy requirements.txt and additional files
COPY requirements.txt .
COPY ./model_converter/model* /var/model_converter/
COPY ./camera_params/intrinsics.xml /var/camera_params/

# install ROS dependencies
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# install project dependencies
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc && \
    echo "export PYTHONPATH=$VIRTUAL_ENV/lib/python3.10/site-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "export ROS_PYTHON_VERSION=3" >> /root/.bashrc && \
    touch /root/setup.sh
# RUN echo "source /rosslam/install/setup.bash" >> /root/.bashrc

# launch terminal
CMD ["/bin/bash"]
