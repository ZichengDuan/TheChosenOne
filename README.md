# The Chosen One: Consistent Characters in Text-to-Image Diffusion Models (Unofficial implementation)

This repository contains the **unofficial** PyTorch implementation of the paper [The Chosen One: Consistent Characters in Text-to-Image Diffusion Models](https://arxiv.org/abs/2311.10093), using the Diffuser framework. 

Shout out to the authors for their great work. 😻😻😻

![Main pipeline](https://github.com/ZichengDuan/TheChosenOne/blob/main/misc/main.png?raw=true)
![Result](https://github.com/ZichengDuan/TheChosenOne/blob/main/misc/result_1.png?raw=true)

(Note that I didn't carefully adjust the parameters for generating these results and they are still good enough.)
## TODO List
- [x] Code release.
- [x] Training instructions.
- [x] Inference instructions.
- [ ] ControlNet support.
- [ ] Local image editing.
- [x] Some visualization results.

## Getting Started

### Installation and Prerequisites
Clone the repository and install the required packages:
```bash
git clone git@github.com:ZichengDuan/TheChosenOne.git
cd TheChosenOne
pip install -r requirements.txt
```
You also need to modify your configuration file in `config/theChosenOne.yaml` to fit your local environment.

### Data backup folder preperation
You need to create a backup data folder to store the initial images generated in the first loop for faster training start up next time if you want to train on the same character again.
This is set up in the configuration file as follows:
``` 
backup_data_dir_root: Your absolute path to the data folder
```

## Run the codes
### Training
```
python main.py
```

### Inference
Simply run:
```
python inference.py
```
The script will load the model you designated in the `inference.py` and your config file.

### Results
TBD.


### Citing the paper
Please always remember to respect the authors and cite their work properly. 🫡
```
@article{avrahami2023chosen,
  title={The Chosen One: Consistent Characters in Text-to-Image Diffusion Models},
  author={Avrahami, Omri and Hertz, Amir and Vinker, Yael and Arar, Moab and Fruchter, Shlomi and Fried, Ohad and Cohen-Or, Daniel and Lischinski, Dani},
  journal={arXiv preprint arXiv:2311.10093},
  year={2023}
}
```