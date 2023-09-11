# Image_Segmentation
This is a project which deals with U-Net model. The data input here is Phantom 4 RTK RGB data over NITK Search site. This same code can also be used for Multispectral, Synthetic Aperture Radar (SAR) data as well, changing the no. of channels. The accuracy found is above 90. This can only be achieved once proper amount of training data is given. Also just in 30 epochs it has already started giving the desired result, running the model for more than the required cycle might overfit the data. For my data these are the no. of epochs and accuracy. This might be different if input data is changed.
The libraries used here are :
1. Tensorflow
2. Matplotlib
3. Numpy
4. cv2
   
I have used the anaconda spyder to run the code in my machine. The libraries and versions can be checked as per the required compatibility.
