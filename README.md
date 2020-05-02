This branch is for project of Advanced Machine Learning course.  
# Abstract
In this project, I experiment with the network designing/modification of DOPE. 
DOPE stands for Deep Object Pose Estimation. It is a 6 DoF pose estimator built 
with approximately 92 conv layers. I wanted to learn more about designing such 
ellaborate neural networks by tyring to play around with the achitecture of 
original DOPE. This project is the result of 1 months work, where I have experimented with 
more than 20 different versions. I could not present my findings or learnings from each of them 
so in this file I present a few of them. First, I present the effects of transfer learning. 
Then I introduce DOPE_2, DOPE_2.1, DOPE_2.2 which are modifications of original DOPE 
architectures. Finally, I present a network called ResNetPose, which is a pose estimator 
like DOPE but built on ResNet34 and has resedual connections with-in each cascade. For each 
presented network, I demostrate an estimation of thier performance. At the end, I conclude with 
my observations and learnings in the discussion section.

# Installation
Use `pip install -r requirements.txt` to install the requirements.

# Downloads
TODO: add weights in public folder in dropbox

# Usage:
Project report: Use `Project.ipynb`
Open in a jupyter-notebook or in google-colab to play around with the file.  

Training: Use `train.py`  
Sample command from terminal: 
`python train.py --network "DOPE_2.2" --outf "dope2.2_meat" --data "/home/mukit/Datasets/fat/single/010_potted_meat_can_16k/" --epoch 60 --featureNet "vgg" --lr 0.0001 --gpuids 1`  
More details of training can be found inside the `train.py` file.
