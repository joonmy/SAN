# Getting Start

## Installation

```
conda create -n gfslt python==3.8.20
conda activate gfslt

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirement.txt
```

## Prepare the Visual Encoder

```
cd model/
python generate_vis_encoder.py
```

## Training

```
bash train.bash
```
