# Yolo-V8-custom-training


YOLOv8 is the latest iteration of the YOLO family of models. YOLO stands for You Only Look Once and these series of models are thus named because of their ability to predict every object present in an image with one forward pass. 
The main distinction introduced by the YOLO models was the framing of the task at hand. The authors of the paper reframed the object detection task as a regression problem (predict the bounding box coordinates) instead of classification.
Setting up your machine for YOLOv8
I have created this GitHub repository which should be sufficient to handle the complete Wildlife Tracking Project.
Download the zip and extract it. Navigate to the folder and install the following libraries and configurations.


1. Setting up Conda Environment
First, we need to create a custom environment for this project. 
conda create -n yolo_env python==3.8


Then, activate this environment.
conda activate yolo_env




2. Installing PyTorch
PyTorch is the underlying framework on top of which Ultralytics is built. PyTorch makes it easy to switch training from CPUs to GPUs. It does a whole lot of other things, but we are going to primarily use it to switch training to GPU. 
We will install PyTorch by running the following command.
conda install pytorch torchvision torchaudio -c pytorch




3. Installing Ultralytics
Previously for using YOLO models, we had to download the YOLO repo and run our code from that directory. Now, we have Ultralytics packaging YOLO models as pip packages. 
We can easily install them by running the following command.
pip install -U ultralytics




4. Selenium
Selenium is a browser automation library. We will be using Selenium to scrape images off the internet. 
pip install selenium




5. Supervision
Supervision is a package that has a lot of repetitive computer vision utilities built into it. We will be using Supervision to create a Tracker Line, which is used to count the number of objects crossing the line. 
Note: I have used supervision==0.3 throughout the project. Upgrading to the latest versions could potentially break the code. 
We can install supervision through pip
pip install supervision==0.3


Now that we've set up the local environment, it's time to visit Narayanan!


Chat with Narayanan
After setting up his local computer for performing Object Tracking, Laxman visited Narayanan’s house. While drinking tea (sourced from the hills of Ooty), they discussed Laxman’s project. 
Laxman has some theoretical understanding of YOLO models, but he lacks coding experience. To familiarize himself with coding with YOLOv8, Narayanan gave him 3 tasks:
Create an object classifier that can look through an image and identify classes present in the image. 

Create an object detector that can parse through a video and draw bounding boxes around objects present in the video.

Create an object segmentation model which can look through an image and segment different objects present in the image. 


Before Laxman could dive into those tasks, he first needed to know the types of tasks YOLOv8 is used for. 


What tasks can YOLOv8 be used for?
YOLOv8 can be used for 3 major computer vision tasks. 


1. Classification
Classification is a simple task in which our model needs to identify and output a single class that is present predominantly in the input image. 
The output of a classification task is a class index and a confidence score. 
In general, classification is useful only when we need to identify if a certain class is present in our input image. Again, one important point to note is the fact that we cannot determine the location of the object, only its presence. 
Creating a classifier is easy with YOLOv8. All we need to do is add the -cls suffix to our desirable model of YOLOv8. 
YOLOv8 comes in different sizes, such as yolov8n (Nano), yolov8s (Small), yolov8m (Medium), yolov8l(Large), and yolov8x (Extra Large).
It is able to perform classification really well because of its pretraining on the ImageNet dataset (a huge dataset containing millions of images).
Now, let's create a classifier with the yolov8n configuration.
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8n-cls.pt')


# Classification result
result = model('SOURCE_PATH')




Exercise 1
Load the classifier version of the yolov8m model and perform classification on the following image. The image is available in the Assets folder from the GitHub repository. 
Hopefully, it was an easy task. Here’s the solution:
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8m-cls.pt')


# Classification result
result = model('../Assets/Bird.jpg', save = True, project = "../Results/")




Classification of Hummingbird, Image by Author




2. Object Detection
Object Detection is the evolution task of Classification. In Object Detection, we need to identify different classes present in the image and detect their exact location. 
The location of such objects is visually shown through Bounding Boxes. 
YOLOv8 models are pretrained on the COCO dataset (another huge image dataset). It can perform Object Detection out of the box.
There is no need for any suffixes.
Here’s an example of performing Object Detection on an image:
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8n.pt')


# Object Detection result
result = model('SOURCE_PATH')




Exercise 2
Load the Object Detection version of the yolov8s model and test it out on the DogVid video available in the Assets folder of the GitHub repository. 
Here’s the solution:
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8n.pt')


# Object Detection result
result = model('../Assets/DogVid.mp4', save = True, project = "../Results/")


The generated video is available in the GitHub repo under Results/ directory.


3. Segmentation
Segmentation is on the next rung after Object Detection. In Object Detection, we found the location of objects and approximated their location through Bounding Boxes. Segmentation is the task in which we identify individual pixels which belong to an object. 
It is much more precise than Object Detection and has a huge range of applications, such as Medical Imaging, Satellite Imaging, etc. 
It is easier to perform Segmentation with YOLOv8. All you need to do is add the -seg suffix.
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8n-seg.pt')


# Segmentation result
result = model('SOURCE_PATH')




Exercise 3
Load the segmentation model and segment the Apples.jpg present in the Assets folder of the GitHub repository.
Here’s the solution:
from ultralytics import YOLO


# Load the model
model = YOLO('yolov8m-seg.pt')


# Segmentation result
result = model('../Assets/Apples.jpg', save = True, project = "../Results/")




Segmentation of Apples, Image by Author




Homestretch
Laxman, after getting hands-on with YOLOv8, is finally ready to work on his Wildlife Tracker. He sets up a face-to-face meeting with Narayanan as quickly as possible.
The day of their meeting is here, and Laxman is excited. Narayanan, in the meantime, created a step-by-step architecture for Laxman to follow.
The guide has every step marked clearly. Laxman takes this architecture home and gets started on building the very first Wildlife Tracker. 
Here’s the architecture:


Custom Tracker Architecture, Image by Author




Step 1: Data Collection
Data Collection is the first stage in any ML project. Remember that the model’s quality depends on the type of data we collect. 
There are many easy ways to collect images, such as the Download All Images Plugin. It could generally make your life easier to download images from the internet, but I have this tendency to make stuff really challenging for myself.
We are going to use the Selenium library to scrape images off the internet. 
Navigate to the 1_Data_Collection directory in the project GitHub repository. You can just run the data_collection.py. This will create an images directory in another subfolder - 2_Data_Annotation. 
I modified the code made by Ivan Goncharov. His code has some deprecated methods, but his tutorial is amazing. It is highly recommended to go through it to understand web scraping with Selenium. 
I am going to explain the changes I made in his code, so you can make modifications if you want to customize it for some other class of objects.
All you need to do to customize to your very own class is to understand the scrape_images method. It takes in three arguments: 
search_term: Object Class you want to scrape e.g., Indian Tiger. We need to remove the space and add + in between. So the search_term becomes indian+tiger
number_of_images: Number of images you want from that object class. 

starting_number: Indexing should not overwrite previously downloaded images. We can set starting_number to add to previously scraped data. Example: If we have 10 images of Tiger (starting with index 0), and we wanted to scrape Lion. We have to set the starting_number to 10. 


After running the script, we should have about 600 images in the 2_Data_Annotation/images folder. Some of these images could be corrupt. We will remove them in upcoming sections.
