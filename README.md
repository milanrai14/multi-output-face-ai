# multi-output-face-ai

Multi-output CNN for simultaneous **age prediction** (regression) and **gender classification** from a single face image. Built with a **VGG16** backbone (transfer learning) and the **Keras Functional API**, and deployed as a REST API using **FastAPI**.

---

## 🧠 Overview

This project demonstrates a **multi-task deep learning model** that takes one face image as input and produces **two outputs** from a **shared convolutional backbone**:

- **Age** — predicted as a continuous number (regression)
- **Gender** — predicted as Male/Female (binary classification)

Rather than training two separate models, both tasks share the same **VGG16** feature extractor, then branch into two independent output heads. This architecture — one input, multiple outputs, shared weights — is only possible using the **Keras Functional API** (not `Sequential`).

---

## 🏗️ Architecture

```
     Input 
        │
        ▼
   VGG16 (pretrained on ImageNet, frozen/fine-tuned)
        │
        ▼
  GlobalAveragePooling2D
        │
        ▼
   Dense(128) + Dropout
        │
   ┌────┴─────┐
   ▼          ▼
Age Head   Gender Head
(Dense→1,   (Dense→1,
 linear)     sigmoid)
   │          │
   ▼          ▼
 Age (yrs)  Gender (M/F)
```

**Key design choices:**
- **VGG16 backbone** — pretrained on ImageNet, used as a fixed/fine-tuned feature extractor instead of training a CNN from scratch. Gives much stronger performance on a modest dataset like UTKFace.
- **Shared trunk, two heads** — one backbone learns general facial features useful for both tasks, reducing redundancy and overfitting compared to training two separate networks.
- **GlobalAveragePooling2D** instead of `Flatten()` — standard practice with pretrained backbones, reduces parameters and overfitting risk.
- **Two-phase training** — first train the new heads with VGG16 frozen, then unfreeze and fine-tune the whole network at a very low learning rate.

---

## 📂 Dataset

**[UTKFace](https://susanqq.github.io/UTKFace/)** — ~20,000 face images labeled with age, gender, and ethnicity directly in the filename (e.g. `25_0_0_20170116174525125.jpg` → age=25, gender=0/male).

- No separate label file needed — labels are parsed from filenames.
- Images are resized to `224x224` and preprocessed with `VGG16.preprocess_input` before training.



> **Note:** Trained `.h5` model files are often too large for a normal git push. Consider using [Git LFS](https://git-lfs.github.com/) or hosting the model on Google Drive / Hugging Face and linking it here instead.

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Model backbone | VGG16 (Keras Applications) |
| Architecture | Keras Functional API |
| Training environment | Google Colab (GPU) |
| Data processing | OpenCV, NumPy, scikit-learn |
| API framework | FastAPI |
| Serving | Uvicorn |

**Python:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    files={"file": open("test_face.jpg", "rb")}
)
print(response.json())
```

**Sample response:**
```json
{
  "predicted_age": 27.4,
  "predicted_gender": "Female",
  "gender_confidence": 0.812
}
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Age MAE | *fill in after training* |
| Gender Accuracy | *fill in after training* |

*(Update this table with your actual evaluation numbers after training.)*

---

## 🔮 Possible Extensions

- Add a **third output head** for ethnicity (also encoded in UTKFace filenames)
- Add **data augmentation** (rotation, flip, zoom) to improve generalization
- Switch data loading to a **`tf.data.Dataset` pipeline** for full-dataset training without RAM crashes
- Containerize the API with **Docker** for easier deployment
- Deploy to a cloud service (Render, Railway, Hugging Face Spaces, AWS, etc.)

---

## 📜 License

This project is open-sourced under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Keras Applications — VGG16](https://keras.io/api/applications/resnet/)
- [FastAPI](https://fastapi.tiangolo.com/)




