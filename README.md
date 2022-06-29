# Smile Detector: Project Overview

Main goal for this project is to build a tool able to recognise if person is smiling or not. Solution can be used in photobooth or in camera app on mobile. 


**Python version:** 3.10.4.  
**Packages:** pandas, numpy, matplotlib, keras.  
**Data resource:** https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
**Repo from**  https://gitlab.com/damianbrzoza/computer_vision_sda
**Resources**
- https://www.intel.com/content/www/us/en/developer/articles/technical/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html
- https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
- https://keras.io/api/applications/
- https://github.com/opencv/opencv/tree/master/data/haarcascades


# Starting Point
This project is based on repo copied from gitlab, which contains image data generator and class that preperaing dataframe and provide some look at the pictures inside dataset.

All images in dataset has the same resolution: 218x178 (height x width). Inside dataframe there is information about bbox but the points doesn't match with pictures. After some 
investidation in kaggle discusion I found out that those are references to the orginal pictures that can be found in the internet. Because this project is time limited I decided to crop each picture to square 178x178 during data preparation. 


# Dataset overview
- 202,599 number of face images of various celebrities
- 10,177 unique identities, but names of identities are not given
- 40 binary attribute annotations per image
- 5 landmark locations

Recommended partitioning of images into training, validation, testing sets. Images:
- 1-162770 are training, 
- 162771-182637 are validation, 
- 182638-202599 are testing

# Data preparation
Dataframe created by Dataset class is concatination off all csv files available on kaggle. For my project i need information about:
- image_id
- smiling attribute
- participation 

Because this data set is large, in this project I'm using only 10_000 records from train data, 2_000 records from validation and test data to reduce time nescassary to train a model. Limit can be change or disabled. 

In PrepareData class, which is subclass of Dataset 



# Model 


# Face recognition 


# Summary 
