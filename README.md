# manydepthformer

Reference Code 

- https://github.com/nianticlabs/manydepth
    
## Overview

Colonoscopy image version

## Pretrained weights and evaluation

You can download weights for some pretrained models here

- [SingleDepth UCL(256x256)]
- [SingleDepth CMT UCL(256x256)]


## Environmental Setting 

Anaconda 

    conda create -n infer_depth python=3.7
    conda activate infer_depth
    pip install numpy
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
    pip install protobuf scipy opencv-python matplotlib scikit-image tensorboardX


VSCode Launch (train, evalute, test)
   [.vscode/launch.json]
  
  [SingleDepth UCL(256x256)]: <https://drive.google.com/drive/folders/1HcgcyPu3WOXJqi2rg6t8AiCQHg1D4Tgq?usp=sharing>
  [SingleDepth CMT UCL(256x256)]: <https://drive.google.com/drive/folders/1bkEqJavbK3DOl6pCDdLQdsono8QnZnPt?usp=sharing>
  [.vscode/launch.json]: <https://github.com/fogfog2/infer_depth/blob/master/.vscode/launch.json>
