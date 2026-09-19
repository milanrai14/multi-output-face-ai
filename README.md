# multi-output-face-ai

Educational **multi-output CNN** demonstrating simultaneous **age prediction** (regression) and **gender classification** from a single face image. Built using a **VGG16** backbone with transfer learning, the **Keras Functional API**, and deployed as a REST API using **FastAPI**.

---

## 🧠 Overview

This project demonstrates a **multi-task deep learning model** that takes one face image as input and produces **two outputs** from a shared convolutional backbone:

* **Age** — predicted as a continuous value (regression)
* **Gender** — predicted as Male/Female (binary classification)

Instead of training two separate models, both tasks share the same **VGG16 feature extractor** and then branch into two independent output heads.

This is a practical example of using the **Keras Functional API** to build a model with:

* One input
* A shared CNN backbone
* Multiple outputs
* Separate output heads
* Different loss functions for different tasks

---

## 🏗️ Architecture

```text
                    Input Image
                        │
                        ▼
              ┌───────────────────┐
              │      VGG16        │
              │ ImageNet Pretrained│
              │ Frozen / Fine-tuned│
              └─────────┬─────────┘
                        │
                        ▼
            GlobalAveragePooling2D
                        │
                        ▼
                  Dense(128)
                        │
                     Dropout
                        │
                 ┌──────┴──────┐
                 ▼             ▼
            Age Head       Gender Head
                 │             │
             Dense(1)       Dense(1)
              Linear          Sigmoid
                 │             │
                 ▼             ▼
            Age (Years)    Gender (M/F)
```

### Key Design Choices

* **VGG16 Backbone**
  Uses VGG16 pretrained on ImageNet as a feature extractor. The convolutional layers learn useful visual features such as edges, textures, shapes, and higher-level patterns.

* **Keras Functional API**
  The Functional API makes it possible to create a model with a shared backbone and multiple output branches.

* **Shared Trunk, Two Heads**
  Both age and gender tasks use the same learned facial features before branching into their respective prediction heads.

* **GlobalAveragePooling2D**
  Converts the spatial feature maps into a compact feature representation while using fewer parameters than a large `Flatten()` layer.

* **Age Regression Head**
  Uses `Dense(1)` with a linear activation to predict a continuous age value.

* **Gender Classification Head**
  Uses `Dense(1)` with a sigmoid activation to produce a probability for binary classification.

* **Two-Phase Training**
  The model can first train the newly added output heads while VGG16 is frozen and then fine-tune selected/all VGG16 layers using a low learning rate.

---

## 📂 Dataset

This project uses the **[UTKFace Dataset](https://susanqq.github.io/UTKFace/)**.

UTKFace contains face images with information about:

* Age
* Gender
* Ethnicity

The labels are encoded directly in the image filenames.

For example:

```text
25_0_0_20170116174525125.jpg
```

The filename represents:

```text
25 → Age
0  → Gender
0  → Ethnicity
```

In this project, the age and gender values are extracted from the filename.

### Image Preprocessing

Images are:

1. Loaded from the dataset
2. Converted to RGB
3. Resized to `224 × 224`
4. Preprocessed using `VGG16.preprocess_input`
5. Passed to the CNN

No separate label file is required because the labels are obtained from the filenames.

---

## 🧩 Multi-Output Model

The model has one input and two outputs:

```python
model = Model(
    inputs=input_layer,
    outputs=[age_output, gender_output]
)
```

The two outputs use different types of prediction:

```text
Age    → Regression
Gender → Binary Classification
```

Therefore, the model can be compiled with separate losses:

```python
model.compile(
    optimizer="adam",
    loss={
        "age_output": "mse",
        "gender_output": "binary_crossentropy"
    },
    metrics={
        "age_output": "mae",
        "gender_output": "accuracy"
    }
)
```

This demonstrates one of the important advantages of the **Keras Functional API**: different outputs can have different layers, losses, and metrics.

---

## ⚙️ Tech Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Deep Learning        | TensorFlow / Keras        |
| CNN Backbone         | VGG16                     |
| Architecture         | Keras Functional API      |
| Transfer Learning    | ImageNet pretrained VGG16 |
| Dataset              | UTKFace                   |
| Image Processing     | OpenCV, PIL               |
| Numerical Computing  | NumPy                     |
| Data Processing      | Pandas                    |
| Model Evaluation     | scikit-learn              |
| Training Environment | Google Colab GPU          |
| API Framework        | FastAPI                   |
| Server               | Uvicorn                   |

---

## 🚀 FastAPI Prediction

After training and saving the model, it can be served using **FastAPI**.

Start the API with:

```bash
uvicorn main:app --reload
```

The API can then receive a face image and return both predictions.

### Python Request Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    files={"file": open("test_face.jpg", "rb")}
)

print(response.json())
```

### Example Response

```json
{
  "predicted_age": 27.4,
  "predicted_gender": "Female",
  "gender_confidence": 0.812
}
```

> **Note:** This is only an example response. Your actual prediction will depend on the trained model and input image.

---

## 📊 Results

Update the following table with your actual evaluation results after training.

| Metric          | Value                    |
| --------------- | ------------------------ |
| Age MAE         | *Fill in after training* |
| Gender Accuracy | *Fill in after training* |

### Understanding the Metrics

**Age MAE (Mean Absolute Error)** measures the average difference between the predicted age and the actual age.

For example:

```text
Actual Age:    25
Predicted Age: 28

Absolute Error = 3 years
```

**Gender Accuracy** represents the percentage of test images for which the predicted gender matches the dataset label.

---

## ⚠️ Disclaimer

This project is created **for educational and learning purposes**, mainly to understand the **Keras Functional API**, transfer learning, and multi-output/multi-task CNN architectures.

The model demonstrates how a single face image can be processed through a shared CNN backbone and produce multiple predictions.

**The predictions may be incorrect and should not be considered fully accurate or reliable.**

Prediction performance can be affected by factors such as:

* Image quality
* Lighting conditions
* Face orientation and pose
* Facial expression
* Occlusion
* Dataset limitations
* Training configuration
* Model architecture
* Distribution differences between training and real-world images

The age and gender predictions are intended to demonstrate the **machine learning pipeline**, not to provide authoritative personal information.

This project is **not intended for medical, biometric, security, identity verification, employment, or other high-stakes decision-making purposes**.

The main purpose of this project is to learn and demonstrate:

* Keras Functional API
* Multi-output neural networks
* Multi-task learning
* Shared CNN backbones
* Multiple output heads
* Transfer learning with VGG16
* Age regression
* Binary classification
* Model fine-tuning
* Image preprocessing
* FastAPI model serving

---

## 📚 What I Learned

Through this project, I explored:

* How the **Keras Functional API** differs from Sequential models
* How to create models with **multiple outputs**
* How to share a pretrained CNN backbone between tasks
* How **VGG16** can be used for transfer learning
* How to freeze and unfreeze layers
* How to perform **feature extraction and fine-tuning**
* How regression and classification can be combined in one model
* How to preprocess images for VGG16
* How to save and load Keras models
* How to expose a trained model through **FastAPI**
* How to send an image to a REST API for prediction

---

## 🔮 Possible Extensions

Future improvements could include:

* Add a **third output head** for ethnicity classification
* Add image augmentation such as rotation, flipping, and zoom
* Experiment with other pretrained architectures such as ResNet or EfficientNet
* Improve the data-loading pipeline using `tf.data.Dataset`
* Experiment with different loss weights for age and gender
* Add better evaluation and visualization
* Add a web interface using Streamlit
* Containerize the API using Docker
* Deploy the API to a cloud platform
* Experiment with more advanced multi-task learning techniques

---

## 📁 Project Structure

A possible project structure is:

```text
multi-output-face-ai/
│
├── notebook/
│   └── multi_output_face_ai.ipynb
│
├── FastAPI/
│   ├── main.py
│   ├── requirements.txt
│   └── model/
│       └── multi_output_face_model.keras
│
├── test_images/
│   └── test_face.jpg
│
├── .gitignore
├── LICENSE
└── README.md
```

> Dataset files and large trained model files should generally not be committed directly to GitHub.

---

## 📦 Large Model Files

Trained `.h5` or `.keras` files can be too large for a normal Git push.

For large model files, consider using:

* [Git LFS](https://git-lfs.github.com/)
* Google Drive
* Hugging Face
* Another suitable model-hosting service

Avoid committing large datasets directly to the repository unless necessary.

---

## 📜 License

This project is open-sourced under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

* [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
* [Keras VGG16 Documentation](https://keras.io/api/applications/vgg/)
* [Keras Functional API Documentation](https://keras.io/guides/functional_api/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [TensorFlow](https://www.tensorflow.org/)
