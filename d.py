import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Docker runner')

    parser.add_argument('-b', dest='actions', action='append_const', const="build")
    parser.add_argument('-s', dest='actions', action='append_const', const="start")
    parser.add_argument('--image', action='store', type=str, default='rosslam-image')
    parser.add_argument('--hostname', action='store', type=str, default='rosslam')
    parser.add_argument('--name', action='store', type=str, default='rosslam-docker-instance')
    parser.add_argument('--cam1_ind', action='store', type=str, default='0')
    parser.add_argument('--cam2_ind', action='store', type=str, default='2')
 
    args = parser.parse_args()

    # commands example
    # sudo docker build -t my-image-name .
    # sudo docker run --hostname rosslam -it --rm --net=host -v .:/rosslam --name my-container-name my-image-name
    if "build" in args.actions:
        cmd = ["docker", "build",
               "-t", args.image,
               "."
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    if "start" in args.actions:
        cmd = ["docker", "run",
               "--rm", "-it",
               "--network", "host",
               "--gpus", "all",
               "--device", "/dev/video" + args.cam1_ind,
               "--device", "/dev/video" + args.cam2_ind,
               "--hostname", args.hostname,
               "-v", ".:/rosslam",
               "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
               "-e", "DISPLAY=$DISPLAY",
               "-e", "QT_X11_NO_MITSHM=1",
               "--name", args.name,
               args.image
        ]
        print(" ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exception:
            print(exception)
