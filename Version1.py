import os
import random
import shutil
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# Set up variables that will help the program know where to find the data in these 
# Specific folders for training, validating, and testing

BASE_PATH = "malaria"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.8 #the ratio of data we want to use for training
VAL_SPLIT = 0.1 #the ratio of data we want to use for validation

dataset_dir = "/Users/yugdv/Malaria-Detection-/malaria"
# This path is specific to the team member's laptop and may need to be updated for different environments

# This function splits all your images into training, validation, and testing groups based on predefined percentages (80/10/10)
# It organizes the images into folders based on their labels (e.g., "infected" and "uninfected") within these groups
# It makes sure the images are ready for use in a ML model

def create_data_splits():
    imagePaths = list(paths.list_images(dataset_dir))
    random.seed(42)
    random.shuffle(imagePaths)
    
    i = int(len(imagePaths) * TRAIN_SPLIT)
    trainPaths = imagePaths[:i]
    testPaths = imagePaths[i:]
    
    i = int(len(trainPaths) * VAL_SPLIT)
    valPaths = trainPaths[:i]
    trainPaths = trainPaths[i:]
    
    datasets = [("training", trainPaths, TRAIN_PATH), 
                ("validation", valPaths, VAL_PATH), 
                ("testing", testPaths, TEST_PATH)]
    
    for (dType, imagePaths, baseOutput) in datasets:
        if not os.path.exists(baseOutput):
            os.makedirs(baseOutput)
        for inputPath in imagePaths:
            filename = os.path.basename(inputPath)
            label = os.path.basename(os.path.dirname(inputPath))
            labelPath = os.path.sep.join([baseOutput, label])
            if not os.path.exists(labelPath):
                os.makedirs(labelPath)
            shutil.copy2(inputPath, os.path.sep.join([labelPath, filename]))

# The function makes the learning rate smaller over time,
# Which helps the model settle into an optimal solution more effectively as it trains

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    return baseLR * (1 - (epoch / float(maxEpochs))) ** 1.0

# Function counts how many images are in the training, validation, and testing datasets
# These numbers help set up the training process

def train_model():
    totalTrain = len(list(paths.list_images(TRAIN_PATH)))
    totalVal = len(list(paths.list_images(VAL_PATH)))
    totalTest = len(list(paths.list_images(TEST_PATH)))

# Creates slightly modified versions of the training images 
#(e.g., rotated, zoomed, shifted) to help the model learn better by seeing the data
    
    trainAug = ImageDataGenerator(rescale=1 / 255.0, rotation_range=20, 
                                  zoom_range=0.05, width_shift_range=0.05,
                                  height_shift_range=0.05, shear_range=0.05,
                                  horizontal_flip=True, fill_mode="nearest")
    
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    
    trainGen = trainAug.flow_from_directory(TRAIN_PATH, class_mode="categorical", 
                                            target_size=(64, 64), color_mode="rgb", 
                                            shuffle=True, batch_size=32)

# Prepares the training images in batches, applies augmentation, and ensures they are ready for the model

    valGen = valAug.flow_from_directory(VAL_PATH, class_mode="categorical", 
                                        target_size=(64, 64), color_mode="rgb", 
                                        shuffle=False, batch_size=32)
    testGen = valAug.flow_from_directory(TEST_PATH, class_mode="categorical", 
                                         target_size=(64, 64), color_mode="rgb", 
                                         shuffle=False, batch_size=32)
    
#T /his segment of the code uses a pre-trained ResNet50 model, then adds layers on top of the base model, 
#such as a pooling layer to simplify the output, and a dense layer with 2 outputs (for 'Infected' and 'Uninfected') 
#and softmax activation (which ensures that the output values are non-negative and sum to 1, making the predictions valid as negative values 
#or values outside the range of 0 to 1 do not make sense in probability). Additionally, the layers are frozen to keep the ResNet50 layers unchanged
#during training, which speeds up learning and helps avoid overfitting

    baseModel = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    model = Sequential()
    model.add(baseModel)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))

    for layer in baseModel.layers:
        layer.trainable = False
    
    opt = SGD(learning_rate=INIT_LR, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Training model...")
    callbacks = [LearningRateScheduler(poly_decay)]
    H = model.fit(
        x=trainGen, steps_per_epoch=totalTrain // 32,
        validation_data=valGen, validation_steps=totalVal // 32,
        epochs=NUM_EPOCHS, callbacks=callbacks)
    
    print("[INFO] Evaluating network...")
    testGen.reset()
    predIdxs = model.predict(x=testGen, steps=(totalTest // 32) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testGen.classes, predIdxs,
                                target_names=["Infected", "Uninfected"]))
    
    return H

# Function helps you see if the model is getting better over time by comparing its loss and 
#accuracy during training and validation, and then saves the graph for you to look at later

def plot_training(H):
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

# This function defines the number of training epochs and the initial learning rate
# It then splits the dataset into training, validation, and test sets using create_data_splits()
# After function trains the model using the train_model() function and stores the training history 
# Finally, it plots the training and validation loss and accuracy over time with plot_training()

if __name__ == "__main__":
    NUM_EPOCHS = 50
    INIT_LR = 1e-4
    create_data_splits()
    history = train_model()
    plot_training(history)
