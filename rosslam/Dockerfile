# Install Python 3.8 and ROS
FROM python:3.8
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
    python3-pip && python3 -m pip install --upgrade pip

# Download and install the CUDA keyring for Ubuntu 22.04
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-5

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda-12.5/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}

# Copy requirements.txt
COPY requirements.txt .
COPY ./model_converter/model.* /var/model_converter/
COPY ./camera_params/intrinsics.xml /var/camera_params/

# install ROS dependencies
RUN python3 -m pip install -r requirements.txt

# install project dependecies
# RUN echo "source /opt/ros/humble/install/setup.bash" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /rosslam/install/setup.bash" >> /root/.bashrc
RUN touch /root/setup.sh

# launch terminal
CMD ["/bin/bash"]