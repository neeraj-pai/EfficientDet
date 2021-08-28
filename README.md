# EfficientDet
This is an implementation of GRADCAM for the [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) implementation by https://github.com/xuannianz/EfficientDet. This can help look at the areas that are getting activated in the various layers to help debug issues.  

## About pretrained weights
* The pretrained EfficientNet weights on imagenet are downloaded from [Callidior/keras-applications/releases](https://github.com/Callidior/keras-applications/releases)
* The pretrained EfficientDet weights on coco are converted from the official release [google/automl](https://github.com/google/automl).

Thanks for their hard work.
This project is released under the Apache License. Please take their licenses into consideration too when use this project.

##build 
Run pip install -r requirements.txt to install the project requirements.
Then run python setup.py build_ext --inplace

## Test
`python3 efficientdet_visualize.py` to test your image by specifying image path and model path there. 

![image1](test/demo.jpg) 
