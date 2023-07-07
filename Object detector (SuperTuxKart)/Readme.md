# Introduction
Implementing a point-based object detection that will predict a dense heatmap of object centers. 
Each "peak" (local maxima) in the heatmap corresponds to a detected object.


# Dataset
Download the dataset and unzip it using the following code:

```python
!wget https://www.cs.utexas.edu/~philkr/supertux_classification_trainval.zip
!wget https://www.cs.utexas.edu/~philkr/supertux_segmentation_trainval.zip

!unzip -q supertux_classification_trainval.zip
!unzip -q supertux_segmentation_trainval.zip
```

# Output

