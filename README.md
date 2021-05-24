# ObjectDetectionGTA5
Implemented object detection using Tensorflow's object detection API and used it in GTA V

The only code needed is in grabscreen.py and vehicleDetection.py .
For the rest of the part, follow the instructions from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
and place the grabscreen.py and vehicleDetection.py files in research\object_detection directory.

![GIF1](https://user-images.githubusercontent.com/38715446/54090873-db9e7700-439e-11e9-8056-ccab8b2f78c1.gif)

![GIF2](https://user-images.githubusercontent.com/38715446/54091194-20c4a800-43a3-11e9-8dd6-6a3a00c85d6a.gif)


SelfDrive.py file in Behavioral Cloning folder is unsed to train a Convolutional neural net with the training images acquired from Udacity's self-driving car simulator.
The model is then saved with the name model3.h5 and is then used by drive.py to connect the simulator to the model.

Now, the simulator sends frames to drive.py which in turn returns the correct steering angle as an output from the model and then the car drives accordingly.


