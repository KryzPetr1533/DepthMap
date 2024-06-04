FROM python:3.8
FROM dustynv/ros:humble-ros-base-l4t-r32.7.1

# cool terminal
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# workdir
WORKDIR /rosslam

# install editors
RUN apt update
RUN apt -y install tmux
RUN apt-get -y update
RUN apt-get -y install vim nano

# install python with pip
RUN apt-get -y install python3-pip
RUN python3 -m pip install --upgrade pip

# install l4v-utils for camera checking 
RUN apt-get -y install v4l-utils

# install ROS dependencies
RUN python3 -m pip install -r requirements.txt

# install project dependecies
RUN echo "source /opt/ros/humble/install/setup.bash" >> ~/.bashrc
# RUN cat /opt/ros/humble/setup.bash >> ~/.bashrc
RUN touch /root/setup.sh

# launch terminal
CMD ["/bin/bash"]