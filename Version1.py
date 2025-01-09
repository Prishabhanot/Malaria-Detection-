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

# Constants
BASE_PATH = "malaria"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Path to the manually downloaded dataset
dataset_dir = "path_to_downloaded_dataset/cell_images"  # Replace with your actual path

# Check if the directory exists and print the paths
if not os.path.exists(dataset_dir):
    print(f"[ERROR] Dataset not found in path {dataset_dir}")
    exit()

# Data Split Function
def create_data_splits():
    imagePaths = list(paths.list_images(dataset_dir))
    random.seed(42)
    random.shuffle(imagePaths)
    
    # Split into train, validation, and test datasets
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

# Learning rate decay function
def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    return baseLR * (1 - (epoch / float(maxEpochs))) ** 1.0

# Model training function
def train_model():
    totalTrain = len(list(paths.list_images(TRAIN_PATH)))
    totalVal = len(list(paths.list_images(VAL_PATH)))
    totalTest = len(list(paths.list_images(TEST_PATH)))
    
    # Data Augmentation
    trainAug = ImageDataGenerator(rescale=1 / 255.0, rotation_range=20, 
                                  zoom_range=0.05, width_shift_range=0.05,
                                  height_shift_range=0.05, shear_range=0.05,
                                  horizontal_flip=True, fill_mode="nearest")
    
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    
    # Data generators for training, validation, and testing
    trainGen = trainAug.flow_from_directory(TRAIN_PATH, class_mode="categorical", 
                                            target_size=(64, 64), color_mode="rgb", 
                                            shuffle=True, batch_size=32)
    valGen = valAug.flow_from_directory(VAL_PATH, class_mode="categorical", 
                                        target_size=(64, 64), color_mode="rgb", 
                                        shuffle=False, batch_size=32)
    testGen = valAug.flow_from_directory(TEST_PATH, class_mode="categorical", 
                                         target_size=(64, 64), color_mode="rgb", 
                                         shuffle=False, batch_size=32)
    
    # Build the ResNet50 model
    baseModel = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    model = Sequential()
    model.add(baseModel)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))  # Output layer for 2 classes (Parasitized, Uninfected)

    # Freeze layers of ResNet50 except the last block
    for layer in baseModel.layers:
        layer.trainable = False
    
    # Compile the model
    opt = SGD(learning_rate=INIT_LR, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    print("[INFO] Training model...")
    callbacks = [LearningRateScheduler(poly_decay)]
    H = model.fit(
        x=trainGen, steps_per_epoch=totalTrain // 32,
        validation_data=valGen, validation_steps=totalVal // 32,
        epochs=NUM_EPOCHS, callbacks=callbacks)
    
    # Evaluate the model
    print("[INFO] Evaluating network...")
    testGen.reset()
    predIdxs = model.predict(x=testGen, steps=(totalTest // 32) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))
    
    return H

# Plot training results
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

# Main script execution
if __name__ == "__main__":
    NUM_EPOCHS = 50  # Total number of epochs
    INIT_LR = 1e-4   # Initial learning rate
    create_data_splits()
    history = train_model()
    plot_training(history)
