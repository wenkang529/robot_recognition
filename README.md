# Robot recognition
## Introduction 
Robot recognition programme has four functions: 
* Face detection  
* Barcode recognition
* Thermometer digits recognition
* Circle dashboard detection

>Language: Python 2  

>Frame: TensorFlow

>OS: Linux

>Version: 0.1

`main.py` can be run directly, in which each function run in its own thread in order to reduce the running time. 

In `main-ros.py`, every function needs the message which is pushed by ros to start. The value of global variable will change after receiving the messages. 
```
global code_flag
global qr_flag
global dash_flag
global digit_flag
global face_flag
```

## Face detect &face align
### mtcnn
mtcnn: Multi-task Cascated Convolution Networks,joint face detection and alignment.It include three stages as following,tree-stage cascated framework  

* stage 1: called proposal network(PNet),it exploit fully convolution network,to obtain the candidate facail windows and their bouding box regression vectors.Then candidates are calibrated based on the estimated bounding box regression vectors.After that,employ non-maximun sunpression(NMS) to merge hightly overlapped candidates.
* stage 2: called Refine network (Rnet) ,just like stage 1,but further rejects a large number of false candidates.
* stage 3: called output network(Onet),this stage is similar to second stage ,but this stage aim to identify face regions with more supervision in particular,the network will output five facial landmarks'positions.

## Face recongnition
### Facenet
* Input:cropped picture rely on mtcnn.
* Output:face feature ,tuple which type is float32 ,number is 128.

### Recognition
* Computing the european distance,some people's face feature have close european distance.

## Barcode recongnition
Using python package [pyzbar](https://pypi.org/project/pyzbar/) to detect both barcode and QR code which set on computers. And then recognizing the information of the code.

## Thermometer digits recognition
Combined [MNIST](http://yann.lecun.com/exdb/mnist/) with [Seven-segment display(ssd)](https://en.wikipedia.org/wiki/Seven-segment_display) to recognize the digits on thermometers. Because of MINIST model aimed to handwritten digits, the accuracy of using this model was not satisfied. So I combined two methods to complished this task. 

## Circle dashboard detection
OpenCV [Hough circle transform](https://docs.opencv.org/3.4/da/d53/tutorial_py_houghcircles.html) was used to detect the circle dashboards in the data center. 

## Result
Each funtion can output the four image point coordinates, while the circle dastection function return the circle center point and radius. These points can represent the relevant location of objects in the image. 

