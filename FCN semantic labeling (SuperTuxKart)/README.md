# Introduction
Improving the Convolutional model for classifying images from SuperTuxKart by trying with:
- Input normalization
- Residual blocks
- Dropout
- Data augmentations (Both geometric and color augmentations)
- Weight regularization
- Early stopping
  
Besides, Train a Fully Convolutional network to classify images from SuperTuxKart.

# Dataset
Download the dataset and unzip it using the following code:

```python
!wget https://www.cs.utexas.edu/~philkr/supertux_classification_trainval.zip
!wget https://www.cs.utexas.edu/~philkr/supertux_segmentation_trainval.zip

!unzip -q supertux_classification_trainval.zip
!unzip -q supertux_segmentation_trainval.zip
```

Visualize the dense dataset:
```python
%run -m utils
```
![dense data](dense_dataset.png)

# Training
Loss and Accuracy during training (CNN):

![train](training_cnn.png)

Loss and Accuracy during training (FCN):

![train](training_fcn.png)

# Output
Original image:

![visualization](vaild_data.png)

Image label:

![visualization](vaild_label.png)

Image label predict:

![visualization](vaild_pred.png)
