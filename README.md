# The Methodological Pitfall of Dataset-Driven Research on Deep Learning: An IoT Example

This repo includes the model implementation and the data for paper: *The Methodological Pitfall of Dataset-Driven Research on Deep Learning: An IoT Example* 

In this paper, we highlight a dangerous pitfall in the state-of-the-art evaluation methodology of deep learning algorithms. It results in deceptively good evaluation outcomes on test datasets, whereas the underlying algorithms remain prone to catastrophic failure in practice.

We use a vehicle detection application to illustrate this pitfall described in our paper. 

## Dataset
Our dataset can be found at: 
[https://uofi.box.com/s/dramtz5j9mhpmo7yx33qd08we646t2oy](https://uofi.box.com/s/dramtz5j9mhpmo7yx33qd08we646t2oy)

Unzip the file and put it in the root directory of `AI_Pitfall`. `pt_data_mustang_development_milcom_noaug` and `pt_mustang_siebel_10-16` are the development dataset. `pt_data_mustang_deployment_milcom_aug` is the deployment dataset.

Each .pt file contains a 2-second data sample. To load a .pt file in PyTorch, use:
```python
import torch
sample = torch.load(FILE_NAME)

label = sample['label'] # vehicle type label
audio_data = sample['data']['shake']['audio'] # audio data, in shape of [1, 10, 1600]
seismic_data = sample['data']['shake']['seismic'] # seismic data, in shape of [1, 10, 20]
```

The file name prefixes in this dataset and their corresponding meta information are listed in the following table:

![alt text](https://raw.githubusercontent.com/leowangx2013/AI_Pitfall/main/imgs/dataset_info.png)


## For Deepsense (the deep learning model):

### Training

Under `AI_Pitfall/deepsense/src`, run:

```
python train.py -dataset train -train_mode original -stage pretrain_classifier
```


### Testing

Under `AI_Pitfall/deepsense/src`, run:
```
python test.py -dataset=TEST_DATASET -stage=pretrain_classifier -model DeepSense -gpu 0 -model_weight ../weights/deepsense_best.pt -batch_size 32
```

For example, to evaluate the trained model on the deployment secenario E:
```
python test.py -dataset=test_scene_E -stage=pretrain_classifier -model DeepSense -gpu 0 -model_weight ../weights/deepsense_best.pt -batch_size 32
```

## For simple model:

### Training

After putting the partitioned data folders to the project directory, run:

```
python simplemodel.py --mode=train
```


### Testing

Under `AI_Pitfall/simplemodel/src`, run:
```
python simplemodel.py --mode=test --scenario=TEST_DATASET
```
where TEST_DATASET are different deployment scenarios with different environment and terrain conditions E-F-G.

For example, to evaluate the trained model on the deployment secenario E:
```
python simplemodel.py --mode=test --scenario=TEST_E

```
