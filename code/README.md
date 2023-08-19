# Code README

## Usage Instructions

This README provides detailed steps to set up and use the code in this repository. The code is designed to be run within a Docker container, which provides an isolated environment for execution.

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- Docker: [Installation Guide](https://docs.docker.com/get-docker/)
- Git: [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### Step 1: Clone the Repository

If you haven't already, clone this repository to your local machine:

```bash
git clone https://github.com/x1y2z3456/capsule-network-for-pcb-defect-inspection.git
cd capsule-network-for-pcb-defect-inspection
```

### Step 2: Run Docker Container

Start a Docker container using the built image and interact with it:

```bash
sudo docker run --name=keras -v /home/user/capsule-network-for-pcb-defect-inspection/:/tmp --network=host -it 105552010/keras:v2.4.0 /bin/bash
```

### Step 3: Run Python Script

Inside the Docker container, you can run the `train.py` script to start the training process:

```bash
python3 train.py
```

### Step 4: Start Jupyter Notebook

Inside the Docker container, for starting a Jupyter Notebook server and access it from your browser, run the following commands:

```bash
bash start_jupyter.bash
```

Copy the URL provided in the terminal and paste it into your browser to access the Jupyter Notebook interface.

### Clean Up

After you're done, you can exit the Docker container by typing `exit`. To stop and remove the container, use the following command:

```bash
sudo docker stop $(docker ps -a -q)
sudo docker rm $(docker ps -a -q)
```

## Troubleshooting

If you encounter any issues during setup or execution, please refer to the documentation or feel free to open an issue in this repository.

## Other Resources
- [Official Example](https://keras.io/zh/examples/cifar10_cnn_capsule/) ( support tensorflow1.15- only )
- [COVID-CAPS](https://github.com/ShahinSHH/COVID-CAPS)
- [New Official Example](https://github.com/keras-team/keras/pull/13620/files/940b724e9befa9389645bf8e6c353646c7fa2974) ( support tensorflow2.3+, I've used this to implement adamW optimizer, check my code if necessary. )
- [Reference Papers](https://hdl.handle.net/11296/45rdc4) ( Check my thesis reference )

## License

This project is licensed under the [MIT License](../LICENSE). See the [LICENSE](../LICENSE) file for details.

---

For any questions or support, please contact [andy345694@gmail.com](mailto:andy345694@gmail.com).
