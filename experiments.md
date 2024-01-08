### 1/8/2024 (3am thank you american airlines)
* 4 epochs
* Starting shape of 3 and ends at 36
* Training on art styles (style_train.csv, style_val.csv)
* Increasing kernel sizes starting from 3 to 7 (for 3 convolution layers)
```
Epoch 1/4, Train Loss: 3.2602, Train Acc: 0.1116, Val Loss: 3.0765, Val Acc: 0.2518
Epoch 2/4, Train Loss: 3.0045, Train Acc: 0.2113, Val Loss: 2.7844, Val Acc: 0.3029
Epoch 3/4, Train Loss: 2.6261, Train Acc: 0.2934, Val Loss: 2.5180, Val Acc: 0.3193
Epoch 4/4, Train Loss: 2.3876, Train Acc: 0.3459, Val Loss: 2.3553, Val Acc: 0.3905
Finished Training
Accuracy: 0.07, Total items: 548
```

### 1/8/2024 (5pm)
* 10 epochs
* Early stopping used
* 42% accuracy using ~10% of the train dataset that is returned from grabbing the style_train csv data and ~10% of the val dataset when grabbing images using the style_val csv data.
* 3 (3x3) kernels of convolution into a linear layer that linearly classifies between 27 classes from and input 64 matrix
* Early Stopping counter to maintain the best model throughout training and terminating if no improvements found after 5 epochs

```
Epoch 1/10, Train Loss: 2.2716, Train Acc: 0.3693, Val Loss: 2.1343, Val Acc: 0.3448
Validation loss decreased (inf --> 2.134348). Saving model ...
Epoch 2/10, Train Loss: 1.7603, Train Acc: 0.4365, Val Loss: 2.1079, Val Acc: 0.3948
Validation loss decreased (2.134348 --> 2.107907). Saving model ...
Epoch 3/10, Train Loss: 1.7227, Train Acc: 0.4389, Val Loss: 2.1794, Val Acc: 0.4022
EarlyStopping counter: 1 out of 5
Epoch 4/10, Train Loss: 1.7139, Train Acc: 0.4360, Val Loss: 2.1478, Val Acc: 0.3559
EarlyStopping counter: 2 out of 5
Epoch 5/10, Train Loss: 1.7012, Train Acc: 0.4350, Val Loss: 2.1996, Val Acc: 0.3930
EarlyStopping counter: 3 out of 5
Epoch 6/10, Train Loss: 1.6807, Train Acc: 0.4380, Val Loss: 2.1261, Val Acc: 0.4096
EarlyStopping counter: 4 out of 5
Epoch 7/10, Train Loss: 1.6751, Train Acc: 0.4389, Val Loss: 2.0800, Val Acc: 0.4069
Validation loss decreased (2.107907 --> 2.080039). Saving model ...
Epoch 8/10, Train Loss: 1.6782, Train Acc: 0.4394, Val Loss: 2.0843, Val Acc: 0.4161
EarlyStopping counter: 1 out of 5
Epoch 9/10, Train Loss: 1.6741, Train Acc: 0.4428, Val Loss: 2.0755, Val Acc: 0.4133
Validation loss decreased (2.080039 --> 2.075490). Saving model ...
Epoch 10/10, Train Loss: 1.6679, Train Acc: 0.4414, Val Loss: 2.0592, Val Acc: 0.4198
Validation loss decreased (2.075490 --> 2.059157). Saving model ...
Finished Training
Accuracy: 0.42, Total items: 1079
```

### Planned Experiment
* 12 epochs
* 5 convolution layers ending at 256 input size to linear layer
* Kernel sizes now go from (3), (5), (5), (7), (7) across the five blocks
* 25% of the data is now being used in both the validation dataset and training dataset (25% of each)
