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
from tensorflow.keras.models import load_model
from pyimagesearch.resnet import ResNet   


ORIG_INPUT_DATASET = "malaria/cell_images"
BASE_PATH = "malaria"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

#validation, and testing splits
def create_data_splits():
    # Verify dataset path exists (idk if we need this)
    if not os.path.exists(ORIG_INPUT_DATASET):
        print(f"[ERROR] Dataset not found at {ORIG_INPUT_DATASET}. Please check the path.")
        exit()
    
    imagePaths = list(paths.list_images(ORIG_INPUT_DATASET))
    random.seed(42)
    random.shuffle(imagePaths)
    
    # Split data into training, validation, and testing
    i = int(len(imagePaths) * TRAIN_SPLIT)
    trainPaths = imagePaths[:i]
    testPaths = imagePaths[i:]
    
    i = int(len(trainPaths) * VAL_SPLIT)
    valPaths = trainPaths[:i]
    trainPaths = trainPaths[i:]
    
    # Create dataset splits
    datasets = [
        ("training", trainPaths, TRAIN_PATH),
        ("validation", valPaths, VAL_PATH),
        ("testing", testPaths, TEST_PATH),
    ]
    
    for (dType, imagePaths, baseOutput) in datasets:
        print(f"[INFO] Building '{dType}' split...")
        if not os.path.exists(baseOutput):
            os.makedirs(baseOutput)
        for inputPath in imagePaths:
            filename = os.path.basename(inputPath)
            label = os.path.basename(os.path.dirname(inputPath))
            labelPath = os.path.sep.join([baseOutput, label])
            if not os.path.exists(labelPath):
                os.makedirs(labelPath)
            shutil.copy2(inputPath, os.path.sep.join([labelPath, filename]))

# Define learning rate schedule
def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    return baseLR * (1 - (epoch / float(maxEpochs))) ** power

# Train and evaluate the model
def train_model():
    # Count total images in each dataset
    totalTrain = len(list(paths.list_images(TRAIN_PATH)))
    totalVal = len(list(paths.list_images(VAL_PATH)))
    totalTest = len(list(paths.list_images(TEST_PATH)))
    
    # Data augmentation
    trainAug = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    
    # Load data generators
    trainGen = trainAug.flow_from_directory(
        TRAIN_PATH, class_mode="categorical", target_size=(64, 64),
        color_mode="rgb", shuffle=True, batch_size=BS)
    valGen = valAug.flow_from_directory(
        VAL_PATH, class_mode="categorical", target_size=(64, 64),
        color_mode="rgb", shuffle=False, batch_size=BS)
    testGen = valAug.flow_from_directory(
        TEST_PATH, class_mode="categorical", target_size=(64, 64),
        color_mode="rgb", shuffle=False, batch_size=BS)
    
    # Build and compile the model
    print("[INFO] Compiling model...")
    model = ResNet.build(64, 64, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg=0.0005)
    opt = SGD(learning_rate=INIT_LR, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Train the model
    print("[INFO] Training model...")
    callbacks = [LearningRateScheduler(poly_decay)]
    H = model.fit(
        x=trainGen, steps_per_epoch=totalTrain // BS,
        validation_data=valGen, validation_steps=totalVal // BS,
        epochs=NUM_EPOCHS, callbacks=callbacks)
    
    # Evaluate the model
    print("[INFO] Evaluating network...")
    testGen.reset()
    predIdxs = model.predict(x=testGen, steps=(totalTest // BS) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))
    
    return H

# Plotting results
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

# Main Script
if __name__ == "__main__":
    # Hyperparameters
    NUM_EPOCHS = 50
    INIT_LR = 1e-1
    BS = 32
    
    create_data_splits()
    history = train_model()
    plot_training(history)
