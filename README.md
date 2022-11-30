# The Methodological Pitfall of Dataset-Driven Research on Deep Learning: An IoT Example

This repo includes the model implementation and the data for paper: *The Methodological Pitfall of Dataset-Driven Research on Deep Learning: An IoT Example* 

In this paper, we highlight a dangerous pitfall in the state-of-the-art evaluation methodology of deep learning algorithms. It results in deceptively good evaluation outcomes on test datasets, whereas the underlying algorithms remain prone to catastrophic failure in practice.

We use a vehicle detection application to illustrate this pitfall described in our paper. 

## Dataset
Our dataset can be found at: 
[https://uofi.box.com/s/6mwoha5g5148cc4vacojrfouyanyzhhx](https://uofi.box.com/s/6mwoha5g5148cc4vacojrfouyanyzhhx)

Each .pt file contains a 2-second data sample. To load a .pt file in PyTorch, use:
```python
import torch
sample = torch.load(FILE_NAME)

label = sample['label'] # vehicle type label
audio_data = sample['data']['shake']['audio'] # audio data, in shape of [1, 10, 1600]
seismic_data = sample['data']['shake']['seismic'] # seismic data, in shape of [1, 10, 20]
```






## For Deepsense (the deep learning model):

1. Training
```
python train.py -dataset=DATASET_NAME -train_mode=original -stage=preain_classifier
```

2. Testing
```
python test.py -dataset=DATASET_NAME -stage=pretrain_classifier -model DeepSense -gpu GPU_ID -model_weight MODEL_WEIGHT_PATH -batch_size BATCH_SIZE
```

## For simple model:
