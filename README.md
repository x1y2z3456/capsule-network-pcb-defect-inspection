# Thesis README

## Description
Welcome to the README for my thesis project. This repository serves as a comprehensive guide to understand the purpose, goals, and technical aspects of my thesis research. Whether you're a fellow researcher, a student, or someone interested in the topic, this README will provide you with valuable insights.

## Table of Contents
- [Introduction](#introduction)
- [Code Reference](#code-reference)
- [Installation](#installation)
- [Docker Images](#docker-images)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this thesis, I explore `"the ability of capsule network to inspect defects of PCB"`. The primary objective is `"making it possible to defect inspection on images of printed circuit boards using a capsule network"`. Throughout this project, I aim to `"the final accuracy, precision, recall rate, and F-score all reached 99.22%"`.
You may download my thesis [here](https://hdl.handle.net/11296/45rdc4).

## Code Reference
The codebase associated with this thesis is available in the `code` directory. You can find detailed information about the code structure, modules, and usage instructions in the [Code Reference](./code/README.md) document.

## Installation
To set up the development environment and run the code, follow these steps:

1. Clone this repository:
   ```shell
   git clone https://github.com/x1y2z3456/capsule-network-for-pcb-defect-inspection.git
   ```

2. Navigate to the project directory:
   ```shell
   cd capsule-network-for-pcb-defect-inspection
   ```

3. Install the required dependencies ( based on Ubuntu ) :
   NVIDIA DRIVER:
   ```shell
   sudo apt install nvidia-driver-510
   # verify nvidia driver
   nvidia-smi
   ```
   Docker:
   ```shell
   sudo apt-get update
   sudo apt-get install ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   echo \
   "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
   "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```
   You may take a look at the [Docker documentation](https://docs.docker.com/engine/install/ubuntu/).
   Tmux (one of my favorite) :
   ```shell
   sudo apt install tmux
   ```
   You may refer to my tmux configuration, rename the file name to .tmux.conf and place it under the home folder.
   ```tmux conf
   # set <prefix> key to C-a
   set-option -g prefix C-a

   # use C-a again to send ctrl-a to inner session
   bind-key C-a send-prefix

   # vi-style controls for copy mode
   setw -g mode-keys vi

   # select and copy like vi in vi-mode
   bind-key -T copy-mode-vi v send -X begin-selection
   bind-key -T copy-mode-vi y send -X copy-selection

   # paste text in vi-mode
   bind p paste-buffer

   # reload settings
   bind-key R source-file ~/.tmux.conf

   # enable mouse draging in sessions
   set -g mouse on
   ```
   After that, you will be able to:
   1. use tmux to split your command window vertically or horizontally, for example, `Ctrl + A + %` or `Crtl + A + "`.
   2. change the window by mouse click.

## Docker Images
This project offers two versions of Docker images, which provide a consistent and isolated environment for running the code. The images can be found on Docker Hub:
- [Tensorflow 1.0](https://hub.docker.com/r/105552010/keras:v2.3.1-rc1)
- [Tensorflow 2.0](https://hub.docker.com/r/105552010/keras:v2.4.0)

To use Docker for this project, you can pull the desired image and run a container:

```shell
sudo docker pull 105552010/keras:v2.4.0
sudo docker run -it 105552010/keras:v2.4.0 /bin/bash
```

## Usage
Once you have completed the installation, you can run model training script in container.

```shell
sudo docker run --name=keras -v /home/user/capsule-network-for-pcb-defect-inspection/:/tmp --network=host -it 105552010/keras:v2.4.0 /bin/bash
cd /tmp
cd 200x200
python3 train_capsnet_latest15-200-full-size-da-densenet121-r8-reduce_lr-r6-r2.py
```

## Contributing
I welcome contributions to this thesis project. If you're interested in improving the code, adding new features, or fixing issues, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the [MIT License](./LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

---

For any questions or collaborations, feel free to contact me at [andy345694@gmail.com](mailto:andy345694@gmail.com).
