# Dependencies
- NumPy
- OpenCV (cv2)
- Matplotlib
- Keras (Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Adam, to_categorical)
```
pip install numpy opencv-python matplotlib keras
```
# Dataset
FER-2013, Kaggle Link: https://www.kaggle.com/datasets/msambare/fer2013
![image](https://github.com/Neuro-Ghost/CNN_EmotionRecognition/assets/104577834/6074a381-bd58-4efe-bb43-2a1ea9c8769f)

# .ipynb and .py scripts
Depending on your enviroment I uploaded .py and .pynb scripts 

# How to Run
  1. Install and import all required dependecies mentioned above.
  
  2. After downloading the dataset make sure the right location is being used when loading the dataset

  3. Run the script emoCNN.py

  4. Make sure the model is saved as HD5 format

  5. Go to emoCNN_CameraTesting run the script

  6. Test different emotions on the camera and see how accurate the model performs

  7. Due to dataset imbalances, Model is expected to perform well on happy, sad, angry, surprise neutral but poorly on disgust and fear.

     It is recommended to use PyCharm IDE when runing the .py scripts

# Face Detection in Live Testing
The face detection is performed using the Haar Cascade classifier provided by OpenCV which is a pre-trained face detection model.

The specific XML file used is 'haarcascade_frontalface_default.xml', which is part of the OpenCV library for detecting frontal faces.

# Potential Application
### Mental Health Monitoring
Aids therapists in tracking patient emotions during therapy sessions, detects stress levels, and contributes to mental health applications.
### Automotive Industry
Improves driver safety through emotion-aware systems that monitor emotional states, alerting drivers to signs of fatigue, stress, or distraction.
# Contributions
Feel free to contriubte by possibly: 
- Modifying CNN architecture
- Pre processing data by introducing a new algorithm
- Fine tuning of augmentation parameters
- Fine tuning of model training parameters

# References
- https://www.analyticsvidhya.com/blog/2021/11/facial-emotion-detection-using-cnn/
- Cakmak, B., & Develi, I. (2023). Convolutional Neural Network-Based Classification of Facial Emotional Expressions and Computational Complexity Analysis. In Proceedings of the 1st International Conference on Frontiers in Academic Research
