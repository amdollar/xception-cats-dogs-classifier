
**Transfer Learning Image Classification: Cats vs. Dogs using Xception**

## Project Description

This project demonstrates **transfer learning** using a pre-trained Xception model from TensorFlow/Keras to classify images as either **cats** or **dogs**. Transfer learning is a technique where a model trained on one large dataset (ImageNet) is reused and fine-tuned for a different but related task, dramatically reducing training time and improving performance on smaller datasets.

**Key Concepts:**
- **Transfer Learning**: Leverages pre-trained weights from Xception (trained on ImageNet) rather than training from scratch
- **Binary Classification**: Classifies images into two classes (cats or dogs) using sigmoid activation
- **Data Augmentation**: Uses ImageDataGenerator for preprocessing and normalization
- **Custom Functional Model**: Builds a hybrid model combining the pre-trained Xception base with custom dense layers
- **Early Stopping**: Implements a custom callback (MyCLRuleMonitor) to halt training when validation performance meets criteria

## Project Contents

- **Notebook**: `11_TransferLearningCatAndDogsXception.ipynb`
- **Dataset**: `cats_and_dogs.zip` (contains train/ and validation/ folders)
- **Model Architecture**: Pre-trained Xception base + custom dense layers (128 → 128 → 1)
- **Input Size**: 128 × 128 × 3 (RGB images)
- **Output**: Binary classification (sigmoid, probability of dog = 1, cat = 0)

## Requirements

**Python Version:** 3.8+ (tested with 3.11/3.12/3.13)

**Dependencies:**
```bash
pip install --upgrade pip
pip install tensorflow numpy pandas pillow matplotlib scikit-learn
```

Or install in one command:
```bash
pip install tensorflow>=2.10 numpy pandas pillow matplotlib scikit-learn
```

## Dataset Setup

1. Extract the dataset:
```python
import shutil
shutil.unpack_archive('cats_and_dogs.zip', 'cats_and_dogs')
```

2. Expected folder structure:
```
cats_and_dogs/
├── train/
│   ├── cats/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── dogs/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    ├── cats/
    │   └── ...
    └── dogs/
        └── ...
```

## Quick Start

### 1. Open and Run the Notebook
```bash
jupyter notebook 11_TransferLearningCatAndDogsXception.ipynb
```
Or open directly in VS Code and run cells sequentially.

### 2. Key Workflow Steps

**Step 1: Import Libraries**
```python
import pandas as pd
import numpy as np
import tensorflow as tf
```

**Step 2: Data Preprocessing**
- Load train/validation data with ImageDataGenerator (rescale 1.0/255.0)
- Target size: 128×128 for efficient inference
- Class mode: binary (cats=0, dogs=1)

**Step 3: Load Pre-trained Xception Model**
```python
xception = tf.keras.applications.xception.Xception(include_top=False)
# Freeze existing weights to preserve learned features
for layer in xception.layers:
    layer.trainable = False
```

**Step 4: Build Custom Functional Model**
- Input: 128×128×3 RGB image
- Xception base (frozen)
- Flatten → Dense(128, relu) → Dense(128, relu) → Dense(1, sigmoid)

**Step 5: Compile and Train**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_image, validation_data=test_image, epochs=100, callbacks=[MyCLRuleMonitor(0.9)])
```

**Step 6: Predict on New Images**
```python
image = tf.keras.preprocessing.image.load_img('your_image.jpg', target_size=(128,128))
image_arr = tf.keras.preprocessing.image.img_to_array(image)
np_img_arr = np.expand_dims(image_arr, axis=0)
probability = model.predict(np_img_arr)
# If probability[0][0] >= 0.5 → Dog; else → Cat
```

## Model Architecture

```
Input (128, 128, 3)
    ↓
Xception (pre-trained, frozen)
    ↓
Flatten
    ↓
Dense(128, relu) [h1]
    ↓
Dense(128, relu) [h2]
    ↓
Dense(1, sigmoid) [output]
    ↓
Output (probability: 0=cat, 1=dog)
```

## Common Issues & Fixes

### Issue 1: "ValueError: Input shape mismatch"
**Cause:** Prediction image size differs from training target size (128×128).

**Fix:** Always use the same target size:
```python
image = tf.keras.preprocessing.image.load_img('image.jpg', target_size=(128, 128))
```

### Issue 2: "truth value of an array is ambiguous"
**Cause:** The MyCLRuleMonitor callback compares numpy arrays or tensors directly without converting to scalar.

**Fix:** Update the callback to safely convert logs to scalars:
```python
class MyCLRuleMonitor(tf.keras.callbacks.Callback):
    def __init__(self, CL, metric_name='accuracy'):
        super().__init__()
        self.CL = float(CL)
        self.metric_name = metric_name
    
    def _to_scalar(self, value):
        if value is None:
            return None
        try:
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            arr = np.asarray(value)
            return float(arr.item() if arr.size == 1 else arr.mean())
        except:
            return None
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        train_val = self._to_scalar(logs.get(self.metric_name))
        val_val = self._to_scalar(logs.get(f'val_{self.metric_name}'))
        if train_val is not None and val_val is not None:
            if (val_val > train_val) and (val_val >= self.CL):
                self.model.stop_training = True
```

### Issue 3: Dataset not found or extraction fails
**Cause:** `cats_and_dogs.zip` not in the working directory.

**Fix:** 
1. Ensure `cats_and_dogs.zip` is in the notebook's directory
2. Run extraction cell explicitly:
```python
import shutil, os
if not os.path.exists('cats_and_dogs'):
    shutil.unpack_archive('cats_and_dogs.zip', 'cats_and_dogs')
    print('Dataset extracted successfully')
else:
    print('Dataset already exists')
```

### Issue 4: Out of Memory (OOM)
**Cause:** Batch size too large or insufficient GPU/RAM.

**Fix:** Reduce batch size in `flow_from_directory`:
```python
train_image = train_image_data.flow_from_directory(..., batch_size=16)  # reduce from 20
test_image = test_image_data.flow_from_directory(..., batch_size=16)
```

### Issue 5: Poor Model Performance
**Cause:** 
- Pre-trained weights are frozen but dataset is very different
- Insufficient training epochs
- Learning rate too high/low

**Fix:**
- Unfreeze some top layers of Xception and fine-tune:
```python
for layer in xception.layers[-20:]:  # unfreeze last 20 layers
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), ...)
```
- Increase epochs or use learning rate scheduling

## Improvements & Extensions

1. **Add Data Augmentation** to the ImageDataGenerator for better generalization:
```python
train_image_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
```

2. **Use a Better Pooling Layer** instead of Flatten for robustness:
```python
x = tf.keras.layers.GlobalAveragePooling2D()(xception(input_layer))
```

3. **Add Dropout** to reduce overfitting:
```python
x2 = tf.keras.layers.Dropout(0.5)(x1)
```

4. **Save and Load Model**:
```python
model.save('cats_dogs_xception.keras')
loaded_model = tf.keras.models.load_model('cats_dogs_xception.keras')
```

5. **Fine-tune Xception** by unfreezing upper layers for better domain adaptation

## Training Tips

- **Initial Training:** Keep Xception frozen (faster, good baseline)
- **Fine-tuning:** Unfreeze top 20-50 layers with very low learning rate (1e-5)
- **Early Stopping:** Use the callback with CL=0.85-0.95 to avoid overfitting
- **Batch Size:** Start with 16-32 for stability
- **Epochs:** 50-100 epochs usually sufficient with pre-trained model

## Expected Results

With the pre-trained Xception model:
- Training accuracy: 95%+
- Validation accuracy: 85-92% (depending on dataset quality)
- Training time: 5-15 minutes per epoch (on CPU) or 1-3 minutes (on GPU)

## File Size & Memory Usage

- Xception base model: ~81 MB
- Full model after training: ~85 MB
- VRAM required: 2-4 GB (GPU) or 8+ GB (CPU recommended)

## References

- [Xception Paper](https://arxiv.org/abs/1610.02357) - Chollet, 2016
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Keras Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)

## License

No specific license provided. Add appropriate license as needed (MIT, Apache 2.0, GPL, etc.).

## Author & Credits

Created for **Simplylearn - AIML Course (Module 4)** as part of Deep Learning and Transfer Learning examples.

---

**Questions or Issues?** Check the troubleshooting section above or refer to the cell comments in the notebook.
