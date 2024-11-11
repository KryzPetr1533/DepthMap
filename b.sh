# currently we build all packages
rosdep install -i --from-path . --rosdistro humble -y
colcon build
set -a
source install/setup.bash