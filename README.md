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

PrepareData Class has a method "generate_set" which is used for converting images and attributes from dataframe into two numpy array: one containing data (images) cropped to 178x178 with RGB colors and second with labels for those photo. Method takes as parameter limit of records that will be converted and which part of dataset should be used(train, valid, test)

Also this class has additional method for saving generated arrays to file and for reading this file.

# Model 
Model is based on InceptionV3 architecture with custom top layer. In this project I tried to predict if person is smiling or not (binary classification). Model will be used in next part of the project. 

I trained two models with the same architecture, only different was that binary_model2 was trained with augmentation. 

binary_model1 - classification report (test data)  
```

              precision    recall  f1-score   support

           0       0.82      0.78      0.80       996
           1       0.79      0.83      0.81      1004

    accuracy                           0.81      2000
   macro avg       0.81      0.81      0.81      2000
weighted avg       0.81      0.81      0.81      2000

```

binary_model2 - classification report (test data)  
```

              precision    recall  f1-score   support

           0       0.85      0.73      0.79       996
           1       0.77      0.87      0.81      1004

    accuracy                           0.80      2000
   macro avg       0.81      0.80      0.80      2000
weighted avg       0.81      0.80      0.80      2000

```


# Face recognition 
For face detection I used openCV build in CascadeClassifier with pretreined weight for face detection (xml files from openCV github repository - link in resources).
Detected face is scaled to 178x178 size for prediction made by my model. I decided to use binary_model1 (trained with augmentation), because during testing I my opinion it works better than other model. 

Frame around face chaning color if person is smiling from blue to green. 

# Summary 
This project was prepared in limited time as a part of my data science bootcamp. Data used for training, validadtion and testing was around 7% of the whole dataset. I decided to used less data to reduce time and processing power required to train a model. 

Model with accuracy around 80% with ballanced data could be improved by many ways. For start it would be good idea to train model with bigger set of images. 

