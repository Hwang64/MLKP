# MLKP
CVPR18 Paper Multi-scale Location-aware Kernel Representation for Object Detection

MLKP is a novel compact, location-aware kernel approximation method to represent object proposals for effective object detection. Our method is among the first which exploits high-order statistics in improving performance of object detection. The significant improvement
over the first-order statistics based counterparts demonstrates the effectiveness of the proposed MLKP.

The code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 

For multi-gpu training, please refer to [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU/)

## Machine configurations

- OS: Linux 14.02
- GPU: TiTan 1080 Ti
- CUDA: version 8.0
- CUDNN: version 5.0

Slight changes may not results instabilities

## PASCAL VOC detection results

   We have re-trained our networks and the results are refreshed as belows:
   
   ### VOC07_Test set results
   
Networks | mAP |aero|bike|bird|boat|bottle| bus| car| cat|chair| cow|table| dog|horse|mbike|person|plant|sheep|sofa|train|tv |
---------|:---:|:--:|:--:|:--:|:--:|:----:|:--:|:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:----:|:---:|:---:|:--:|:---:|:-:|
  VGG16  | 78.4|80.4|83.0|77.6|70.0| 71.8 |84.2|87.5|86.7| 67.0|83.1| 70.3|84.9| 85.5| 81.9| 79.2 | 52.6| 79.7|79.6|81.7|81.4|     
  ResNet | 81.0|80.3|87.1|80.8|73.5| 71.6 |86.0|88.4|88.8| 66.9|86.2| 72.8|88.7| 87.4| 86.7| 84.3 | 56.7| 84.9|81.0|86.7|81.7|   

   ### VOC12_Test set results
   
Networks | mAP |aero|bike|bird|boat|bottle| bus| car| cat|chair| cow|table| dog|horse|mbike|person|plant|sheep|sofa|train|tv |
---------|:---:|:--:|:--:|:--:|:--:|:----:|:--:|:--:|:--:|:---:|:--:|:---:|:--:|:---:|:---:|:----:|:---:|:---:|:--:|:---:|:-:|
  VGG16  | 75.5|86.4|83.4|78.2|60.5| 57.9 |80.6|79.5|91.2| 56.4|81.0| 58.6|91.3| 84.4| 84.3| 83.5 | 56.5|77.8|67.5|83.9|67.4|
  ResNet | 78.0|87.2|85.6|79.7|67.3| 63.3 |81.2|82.0|92.9| 60.2|82.1| 61.0|91.2| 84.7| 86.6| 85.5 | 60.6|80.8|69.5|85.8|72.4|
  
  Results can be found at [VGG16](http://host.robots.ox.ac.uk:8080/anonymous/TENHEH.html) and [ResNet](http://host.robots.ox.ac.uk:8080/anonymous/NF0WFQ.html)
  
## MS COCO detection results

Networks | Avg.Precision,IOU: | Avg.Precision,Area: |  Avg.Recal,#Det:  |    Avg.Recal,Area:  | 
|--------|:------------------:|:-------------------:|:-----------------:|:-------------------:|
|        |0.5:0.95  0.50  0.75| Small   Med.  Large |   1    10     100 | Small   Med.  Large |
  VGG16  |  26.9    48.4  26.9|  8.6    29.2   41.1 | 25.6  37.9   38.9 |  16.0   44.1   59.0 |
  ResNet |
## MLKP Installation 

0. Clone the RON repository
    ```
    git clone https://github.com/HIT-CS-HWang/MLKP.git
    ```
1. Build Caffe and pycaffe

    ```

    ```

2. Build the Cython modules
    ```
    cd $MLKP_ROOT/lib
    make
    ```
    
3. installation for training and testing models on PASCAL VOC dataset

    3.0 The PASCAL VOC dataset has the basic structure:
    
        $VOCdevkit/                           # development kit
        $VOCdevkit/VOCcode/                   # VOC utility code
        $VOCdevkit/VOC2007                    # image sets, annotations, etc.
        
    3.1 Create symlinks for the PASCAL VOC dataset
    
        cd $MLKP_ROOT/data
        ln -s $VOCdevkit VOCdevkit2007
        ln -s $VOCdevkit VOCdevkit2012
    
    For more details, please refer to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 

 4. Test with PASCAL VOC dataset

     We provide PASCAL VOC 2007 pretrained models based on VGG16 and ResNet, please download the models manully from [BaiduYun](https://pan.baidu.com/s/1HgxsixN674ZfGE-9lm77KQ) and put them in `$MLKP_ROOT/output/`
   
     4.0 Test VOC07 using VGG16 network
      
      python ./tools/test_net.py --gpu 0 --def models/VGG16/test.prototxt --net output/VGG16_voc07_test.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
     
      #### The final results of the model is mAP=78.4%
   
     4.1 Test VOC07 using ResNet-101 network   
      
      python ./tools/test_net.py --gpu 0 --def models/ResNet/test.prototxt --net output/ResNet_voc07_test.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
      
      #### The final results of the model is mAP=81.0%
  
 5. Train with PASCAL VOC dataset
    
     Please download ImageNet-pretrained models first and put them into `$data/ImageNet_models`.
   
     5.0 Train using single GPU
   
      ```
      python ./tools/train_net.py --gpu 0 --solver models/VGG16/solver.prototxt --weights data/ImageNet_models/VGG16.v2.caffemodel --imdb voc_2007_trainval+voc_2012_trainval --cfg experiments/cfgs/faster_rcnn_end2end.yml    
      ```
   
     5.1 Train using mutli-GPUs
   
      ```
      python ./tools/train_net_multi_gpu.py --gpu 0,1,2,3 --solver models/VGG16/solver.prototxt --weights data/ImageNet_models/VGG16.v2.caffemodel --imdb voc_2007_trainval+voc_2012_trainval --cfg experiments/cfgs/faster_rcnn_end2end.yml        
      ```
