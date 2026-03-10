# X-Ray Bone Fracture Detection Using CNN and OpenCV

---
## What This Project Does

This is a deep learning system that looks at bone X-ray images and tells you whether a fracture is present or not. The model was trained on over 8,800 X-ray images and deployed through a Gradio web interface where you can either upload an image or use your webcam in real time.

The idea came from a simple observation — radiologists in busy hospitals or rural clinics often deal with a high volume of scans, and even a small assistive tool that flags potential fractures can save time and reduce the chance of a missed diagnosis. This project is not a replacement for a doctor, but it works reasonably well as a second-opinion tool.

---

## Results

The model was evaluated on 506 held-out test images that were never seen during training.

| Metric | Score |
|--------|-------|
| Test Accuracy | 97.6% |
| F1 Score | 0.977 |
| Test Samples | 506 images |
| Training Images | 8,800+ |

```
              precision    recall  f1-score   support

      Normal     0.97      0.98      0.978       253
   Fractured     0.98      0.97      0.976       253

    accuracy                         0.976       506
   macro avg     0.975     0.975     0.977       506
```

## Model Architecture

The model is a custom CNN built from scratch — no pretrained weights, no transfer learning. The decision to avoid transfer learning was deliberate: the goal was to understand what a network trained purely on this specific domain learns, and to keep the architecture interpretable.

Input (224 x 224 x 3)
    Conv2D(32) + ReLU + MaxPooling
    Conv2D(64) + ReLU + MaxPooling
    Conv2D(128) + ReLU + MaxPooling
    Flatten
    Dense(256) + ReLU
    Dropout(0.5)
    Dense(1) + Sigmoid
Output: Fractured / Normal with confidence score


## Tech Stack

- TensorFlow / Keras for model building and training
- OpenCV for image preprocessing and webcam input
- Gradio for the web interface
- Scikit-learn for evaluation metrics
- Jupyter Notebook as the development environment

## Dataset

- Source: Kaggle Bone Fracture Detection Dataset
- Total images: 8,800+ X-rays across two classes (fractured / normal)
- Train / test split: 80% training, 20% testing
- Augmentation applied: rotation, zoom, horizontal flip, shear

Augmentation was important here because medical imaging datasets tend to be smaller than general-purpose ones. Without it, the model would likely memorise the training data rather than learning patterns that generalise.

## How to Run

Clone the repository:
bash
git clone https://github.com/Pavanv2030/X-Ray-Bone-Fracture-Detection-Using-CNN-and-OpenCV.git
cd X-Ray-Bone-Fracture-Detection-Using-CNN-and-OpenCV

Install dependencies:
bash
pip install -r requirements.txt

Download the trained model — GitHub has a file size limit, so the model is hosted on Google Drive separately:

https://drive.google.com/file/d/1zsrpa9R8g0_fiPf1qAdFXh8I9uUSHU2J/view?usp=drive_link

Place the downloaded `.h5` file in the root folder, then run:
bash
jupyter notebook "opencv and cnn final.ipynb"

---

## Project Structure

```
X-Ray-Bone-Fracture-Detection-Using-CNN-and-OpenCV/
|
|-- opencv and cnn final.ipynb     # training, evaluation, and Gradio UI
|-- requirements.txt
|-- README.md
|-- screenshots/                   # UI screenshots
|-- bone_fracture_model.h5         # trained model (download from Drive link above)
```

---

## Training Decisions

**EarlyStopping**

Training stops automatically when validation loss stops improving, with a patience of 5 epochs. The True flag ensures the saved model is always the best checkpoint rather than the final epoch.

python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

**Dropout**

A rate of 0.5 was applied before the output layer. With a dataset of around 8,800 images, the model can overfit fairly easily — dropout forces the network to not rely on any single set of neurons, which helped close the gap between training and validation accuracy.
python
Dropout(0.5)

**Data Augmentation**

python
ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    shear_range=0.15
)
These transforms were chosen to simulate realistic variation in how X-rays are captured — slight rotations, different zoom levels, and orientation differences are all common in real clinical settings.

---

## What Could Be Improved

A few additions that would make this more useful in practice:

- Fracture localization using YOLOv8, moving from a binary yes/no output to actually drawing a box around where the fracture is
- An LLM-generated summary report based on the prediction confidence and location
- A live deployment on HuggingFace Spaces so anyone can test it without setting up anything locally

---

## Disclaimer

This project was built for learning and research purposes. It is not a certified medical device and should not be used as a substitute for diagnosis by a qualified radiologist.

---

## Author

Pavan V — [GitHub](https://github.com/Pavanv2030)
