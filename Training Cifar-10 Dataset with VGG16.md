## Training Cifar-10 Dataset with VGG16

### Environment:

* python 3.7
* opencv-contrib-python
* Matplotlib 3.1.1
* UI_framework : pyqt5(5.15.1)



### Running Code

* in hw1 dir, use

  ```
  python main.py
  ```

  

​		can see the user interface

<img src="C:\Users\CaramelYo\AppData\Roaming\Typora\typora-user-images\image-20211119174403983.png" alt="image-20211119174403983" style="zoom:50%;" />

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

  ​				<img src="C:\Users\CaramelYo\AppData\Roaming\Typora\typora-user-images\image-20211119180552870.png" alt="image-20211119180552870" style="zoom:50%;" />

1. Show random 9 images in Cifar10 training dataset and their labels 
2. Show the hyperparameters of the network
3. Show the architecture of network
4. Sow images of accuracy and loss
5. input number then show the image and Probability with Histogram of the corresponded data