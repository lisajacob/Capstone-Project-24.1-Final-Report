
![render0006](https://github.com/user-attachments/assets/eb9c8e1c-faaa-471e-97d9-1b3075a3e7e6)
### Classification of Lunar Rocks

**Lisa Jacob**

#### Executive summary
The AI-Based Semantic Segmentation of Lunar Rocks project aims to leverage advanced artificial intelligence and machine learning techniques to enhance the analysis of lunar rock samples. By employing semantic segmentation, this project will automatically classify and delineate different rock types and features from high-resolution lunar imagery, providing valuable insights for scientific research and exploration.

Objectives

Automated Rock Classification: Develop a deep learning model to accurately identify and categorize various types of lunar rocks and mineral compositions from images captured by lunar missions.
Detailed Feature Extraction: Segment and annotate specific features of lunar rocks, such as texture, structure, and composition, to support geological analysis and mission planning.
Enhanced Data Interpretation: Improve the efficiency and accuracy of data processing from lunar missions, enabling faster and more reliable interpretations of rock samples.

Key Technologies and Applications

Deep Learning Models: Utilize Convolutional Neural Networks (CNNs) for high-precision semantic segmentation.
High-Resolution Imagery: Analyze images from lunar rovers or orbiters with high spatial resolution to ensure detailed and accurate rock classification.
Data Augmentation: Implement techniques to enhance model performance and generalizability using synthetic data and augmentation strategies.

Impact and Benefits

Scientific Advancement: Facilitate more in-depth geological studies of the Moon’s surface, aiding in the understanding of its formation and evolution.
Mission Efficiency: Streamline the analysis of lunar rock samples, reducing the time and resources required for manual examination.
Exploration Support: Provide critical data to guide future lunar exploration missions, including site selection and resource assessment.

#### Rationale
One of the goals of computer vision is to give machines the ability to interpret visual scenes. For space robotics, the lunar lander and the rover has to be able to autonomously navigate. Thus its ability to understand the rock size-frequency distribution (RSFD) on lunar surfaces is important. Large rocks within a landing site represent potential hazards to landers as well as navigational threat to rovers. 

#### Research Question
The research for this Capstone project focuses on Semantic Segmentation of rocks on the lunar surface with the ultimate goal to distinguish between small rocks, large rocks, small craters, and large craters on the lunar surface. 

Semantic Segmentation
Semantic segmentation is a deep learning technique that labels every pixel in an image to identify and categorize distinct elements, such as vehicles, pedestrians, and road signs, which is crucial for applications like autonomous driving.

Image Augmentation
Image augmentation creates new training examples by slightly modifying existing images, such as adjusting brightness or cropping.

In semantic segmentation, both input images and output masks must undergo the same transformations. Augmentations fall into two types: pixel-level and spatial-level. Pixel-level augmentations alter image pixel values without affecting the mask, like changing brightness. Spatial-level augmentations, such as rotation or cropping, modify both the image and the mask to keep them aligned.

During training, augmentations are applied with a probability less than 100% to include original images. For large datasets, use basic augmentations with a low probability (10-30%). For small datasets, use more aggressive augmentations (40-50%) to prevent overfitting.

#### Data Sources
Kaggle dataset [Artificial Lunar Rocky Landscape](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)

The dataset currently contains 9,766 realistic renders of rocky lunar landscapes, and their segmented equivalents (the 4 classes are the sky, surface, smaller rocks, and larger rocks). A table of bounding boxes for all larger rocks and processed, cleaned-up ground truth images are also provided. For the project, I used the render, and clean folders to train, test, and validate the model. For additional information and guidelines on how to use the dataset effectively, refer to [Understanding and Using the Dataset](https://www.kaggle.com/code/romainpessia/understanding-and-using-the-dataset-wip) kernel.

#### Methodology

Data Collection: Gathered and preprocessed lunar rock images from the Data Source described above.
Model Development: Train and validate deep learning models using a combination of real and synthetic data to achieve high segmentation accuracy. Utilized PyTorch, and Albumentations libraries.
Integration and Testing: Deploy the models in a test environment to evaluate performance and make iterative improvements.
Deployment: Implement the model for operational use in analyzing lunar rock samples and integrate it with existing data analysis workflows.

For this project, I will be using ResNeXt, a Convolutional Neural Network (CNN) architecture developed by Microsoft Research and introduced in 2017 in the paper titled Aggregated Residual Transformations for Deep Neural Networks. The project will utilize PyTorch, which includes model builders in the Torchvision library to instantiate a ResNeXt model, specifically resnext50_32x4d.


#### Results
Model evaluation on a test sample resulted in:
- Median Pixel Accuracy: 98.72%
- Median IoU: 65.41%

Successful Prediction of Lunar rocks

![prediction1](https://github.com/user-attachments/assets/26bddc85-eaf0-4e8f-89df-8c6635eba7bb)

#### Next steps
1. Determine which augmentations would result in the most accurate results. 
2. Comparison of model performance with different deep learning architectures.
 - Segnet
 - Unet
 - DeepLab Series
3. Provide a streamlit front end for user to select the files and to present the results

#### Outline of project
The project was developed using Google Colab to leverage GPU power for handling large images. For image processing, I created utility functions within a lunar_image_processor library. This library facilitates the creation of necessary folders, and the splitting of data into test, validation, and training sets. It also enables resizing images into smaller patches. To run the code, this module should be imported into the Google Colab environment.
- [Preprocessing and folder creation](https://github.com/lisajacob/Capstone-Project-24.1-Final-Report/tree/main/lunar_image_processor)
- [Exploratory Data Analysis](https://github.com/lisajacob/Capstone-Project-24.1-Final-Report/blob/main/EDA.ipynb)
- [Model development and Findings](https://github.com/lisajacob/Capstone-Project-24.1-Final-Report/blob/main/LunarModels.ipynb)

##### Contact and Further Information
For further information, send an email to lisa.jacob@gmail.com
