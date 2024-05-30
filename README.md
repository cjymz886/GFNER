# GFNER
GFNER: A Unified Global Feature-aware Framework for Flat and Nested Named Entity Recognition

## Setup conda enviroment

- pip install transformers==4.17.0</br>
- pip install bert4keras==0.11.0</br>
- pip install tensorflow==2.2<br>
- conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge</br>

## How to use
```bash
git clone git@github.com:cjymz886/GFNER.git
cd GFNER
python run.py --mode train --config_name genia
```

## Prediction Mode and Primary Mode
The comparison of prediction mode and primary mode:

![primary_prediction](pic/table_fig152.png)

## data
```
contain CoNLL2003,GENIA,weibo dataset, except ACE2005 since no right to share
```
