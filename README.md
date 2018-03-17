# MLKP
CVPR18 Paper Multi-scale Location-aware Kernel Representation for Object Detection

MLKP is a novel compact, location-aware kernel approximation method to represent object proposals for effective object detection. Our method is among the first which exploits high-order statistics in improving performance of object detection. The significant improvement
over the first-order statistics based counterparts demonstrates the effectiveness of the proposed MLKP.

The code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 

For multi-gpu training, please refer to [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU/)

### PASCAL VOC detection results


### MS COCO detection results


### MLKP Installation 

0. Clone the RON repository
    ```
    git clone https://github.com/HIT-CS-HWang/MLKP.git

    ```
1. Build Caffe and pycaffe

    ```

    ```

2. Build the Cython modules
    ```
    cd $RON_ROOT/lib
    make
    ```
    
3. installation for training and testing models on PASCAL VOC dataset

    3.0 The PASCAL VOC dataset has the basic structure:
    
        $VOCdevkit/                           # development kit
        $VOCdevkit/VOCcode/                   # VOC utility code
        $VOCdevkit/VOC2007                    # image sets, annotations, etc.
        
    3.1 Create symlinks for the PASCAL VOC dataset
    
        cd $RON_ROOT/data
        ln -s $VOCdevkit VOCdevkit2007
        ln -s $VOCdevkit VOCdevkit2012

4. Test with PASCAL VOC dataset

   [BaiduYun](https://pan.baidu.com/s/1HgxsixN674ZfGE-9lm77KQ)

5. Train with PASCAL VOC dataset
