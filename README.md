Bird or Drone 🐦🚁

An experimental computer vision project for bird and aerial-object classification.

Bird or Drone is an early-stage project exploring how computer vision and YOLO-based object detection can help distinguish birds from other objects in images and video.

The project is a work in progress. It is shared to encourage learning, experimentation, and community contributions.

🎯 Project Goals

- Explore object detection using Python, OpenCV, and YOLO.
- Process video files and visualize detected objects.
- Experiment with distinguishing birds from other aerial objects.
- Encourage improvements to the model, software, and future hardware integration.

⚠️ Current Limitations

This project is experimental and is not a reliable drone detection or threat identification system.

- The current "main.py" uses the general-purpose YOLOv8n model.
- The current code maps the "airplane" and "kite" classes to the "drone" label. These objects are not necessarily drones.
- The current implementation processes a video file named "video.mp4"; live camera input is not implemented in this script.
- Performance depends on image quality, object size, lighting, model capabilities, and available computing resources.
- The project has not been validated for safety-critical or operational use.

🛠️ Requirements

- Python 3.10 or 3.11 recommended.
- A computer running Windows or Linux.
- Python packages listed in "requirements.txt".
- A video file for testing.

Hardware: A dedicated NVIDIA GPU is not necessarily required for basic inference. A compatible GPU can improve performance, while training a model can require substantially more computing resources. Actual performance depends on the hardware and configuration.

🚀 Installation

1. Clone the repository

git clone https://github.com/ikimodev/drone-or-bird.git
cd drone-or-bird

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

For GPU acceleration, install a compatible PyTorch configuration for your hardware and CUDA environment. GPU support is optional for basic inference.

▶️ Run the Project

1. Place a test video named "video.mp4" in the repository root directory.
2. Activate your virtual environment.
3. Run:

python main.py

The program attempts to detect objects in the video and display labels on the frames.

Press "q" to close the video window.

Note: The current script uses "yolov8n.pt". It does not automatically use "drone_model.pt". Model selection and class mapping need further development for meaningful drone-versus-bird classification.

🧠 Model Training

The repository includes "train.py" for experimenting with YOLO model training and Roboflow datasets.

Training requires a correctly configured dataset, valid Roboflow credentials, and suitable computing resources. Review and configure the training script before running it. Never commit API keys, passwords, or other credentials to the repository.

Training results depend on dataset quality, class balance, annotations, and evaluation methodology.

🗺️ Roadmap

- [ ] Improve drone-versus-bird classification.
- [ ] Evaluate performance on independent test data.
- [ ] Add clear configuration for model selection.
- [ ] Improve error handling and installation documentation.
- [ ] Explore live camera support.
- [ ] Explore optional hardware integration.
- [ ] Document performance, limitations, and reproducible test results.

🤝 Contributing

Contributions and constructive feedback are welcome.

You can help with:

- Python development and computer vision.
- Model evaluation and dataset quality.
- Software testing and documentation.
- Electronics and future hardware prototypes.

Please describe changes clearly and include test results when possible.

🛡️ Responsible Use

This is an educational research project. Its predictions may be incorrect or incomplete.

Do not use its output as the sole basis for safety-critical decisions. Respect privacy, applicable laws, and official instructions. Do not publish sensitive information about real-world security incidents.

📜 License

A license has not yet been specified. Reuse and redistribution permissions should be clarified by adding an appropriate "LICENSE" file.

👨‍💻 Project

GitHub: https://github.com/ikimodev/drone-or-bird

Built as an ongoing learning and open-collaboration project. Contributions that improve its accuracy, reliability, and responsible use are welcome.
