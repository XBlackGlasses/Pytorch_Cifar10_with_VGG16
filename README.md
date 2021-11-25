## Training Cifar-10 Dataset with VGG16

### Environment:

* python 3.7
* opencv-contrib-python 3.4.2.17
  ```
  pip install opencv-contrib-python==3.4.2.17 
  ```
* Matplotlib 3.1.1
  ```
  pip install matplotlib==3.1.1
  ```
* UI_framework : pyqt5(5.15.1)
  ```
  pip install PyQt5==5.15.1
  ```
* pytorch 1.10.0 、 cuda 11.3
  ```
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```


### Running Code

* in hw1 dir, use

  ```
  python main.py
  ```

  

​		can see the user interface

![](https://i.imgur.com/XG65M5o.png)

​		correspond to function below :

| Image Processing                                             | Image Smoothing                                              | Edge Detection                                               | Transforms:                                                  |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.1 Load Image File <br />1.2 Color Separation <br />1.4 Blending<br />1.3 Color Transformation | 2.1 Gaussian blur<br />2.2. Bilateral filter <br />2.3  Median filter | 3.1  Gaussian Blur<br />3.2  Sobel X<br />3.3  Sobel Y<br />3.4  Magnitude | 4.1  Resize<br />4.2  Translation<br />4.3 Rotation, Scaling<br />4.4 Shearing |



* In network dir, use

  ```
  python vgg.py 
  ```

  to training VGG16 network.  We have pretrained model *model.py* can use directly.

  use

  ```
  python main.py
  ```

  can see the user interface

  ​				![](https://i.imgur.com/5OajXHf.png)

1. Show random 9 images in Cifar10 training dataset and their labels 
2. Show the hyperparameters of the network
3. Show the architecture of network
4. Sow images of accuracy and loss
5. input number then show the image and Probability with Histogram of the corresponded data
