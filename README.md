Malaria continues to be one of the most pressing global health challenges, particularly in tropical and subtropical regions. Timely diagnosis and intervention are incredibly important in healthcare settings to reduce its impact.  By developing models that can autonomously detect malaria-infected cells from medical images, healthcare professionals can get faster and more accurate diagnoses. Through the integration of image data augmentation, deep learning models like ResNet50, and data management practices, the goal is to equip medical practitioners with tools that enable rapid and precise malaria detection. The model uses a pre-trained ResNet50 network for classification and is fine-tuned for classifying images as either "Infected" or "Uninfected."
 
The medical dataset inputted has been retrived from Kaggle. 

**Features**

**Data Preparation:** Splits the dataset into training, validation, and test sets.
**Image Augmentation:**  Applies various transformations (e.g., rotation, zoom, shift) to training images to improve model generalization.
**Deep Learning Model:** Fine-tunes the pre-trained ResNet50 model for image classification.
**Learning Rate Scheduler:** Decreases the learning rate during training for better model convergence.
**Model Evaluation:** Prints a classification report showing how well the model performs on the test set.
**Training Visualization:** Plots loss and accuracy curves over time for both training and validation sets.

The code outputs the model's performance over each epoch, with AUC (area under curve) scores and a plot that tracks how the training and validation losses, as well as AUC, change over time, helping you see how the model improves and when early stopping kicks in. For further information AUC=0.5 means the model is no better than random guessing, AUC=1 means the model perfectly distinguishes between the classes, and AUC <0.5 suggests the model is performing worse than random guessing.
