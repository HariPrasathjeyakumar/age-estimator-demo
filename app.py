# app.py
import os
import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from mtcnn import MTCNN

MODEL_PATH = "best_soft_label_model.keras"
GDRIVE_FILE_ID = "1oN5aI1HHgZNga2qZ5ADDXzQBoW0bI8U-"

# ============================================================
# MC DROPOUT PATCHING & CACHED MODEL LOADING
# ============================================================
def patch_mc_dropout(model_obj):
    """Recursively patches Dropout layers to remain active during inference."""
    patched_count = 0

    def _recursive_patch(container):
        nonlocal patched_count
        layers = getattr(container, 'layers', getattr(container, 'submodules', []))
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                if not getattr(layer, '_mc_patched', False):
                    orig_call = layer.call
                    layer.call = lambda inputs, *args, _orig=orig_call, **kwargs: _orig(inputs, training=True)
                    layer._mc_patched = True
                    patched_count += 1
            if hasattr(layer, 'layers') or hasattr(layer, 'submodules'):
                _recursive_patch(layer)

    _recursive_patch(model_obj)
    return patched_count

@st.cache_resource
def load_model_and_verify():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    num_patched = patch_mc_dropout(loaded_model)

    # Diagnostic pass with float32 unscaled range [0, 255]
    dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32) * 255.0
    age_bins = np.arange(101)
    test_preds = [
        float(np.sum(age_bins * loaded_model(dummy_input, training=False).numpy()[0]))
        for _ in range(5)
    ]
    startup_variance = float(np.var(test_preds))

    return loaded_model, num_patched, startup_variance

@st.cache_resource
def load_detector():
    return MTCNN()

model, patched_layers_count, startup_var = load_model_and_verify()
detector = load_detector()

# ============================================================
# FACE DETECTION & STANDARDIZED PREPROCESSING
# ============================================================
def crop_and_align_face(image_pil, target_size=(256, 256)):
    # Standard RGB pipeline matching Kaggle
    img_array = np.array(image_pil.convert('RGB'))
    results = detector.detect_faces(img_array)

    if not results:
        return None, None, ["⚠️ No face detected — please upload a clearer face image"]

    best_face = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)

    pad_x, pad_y = int(w * 0.20), int(h * 0.20)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img_array.shape[1], x + w + pad_x), min(img_array.shape[0], y + h + pad_y)

    face_crop = img_array[y1:y2, x1:x2]
    resized = cv2.resize(face_crop, target_size)
    
    # Range [0.0, 255.0] float32 to let internal EfficientNet Rescaling layer operate properly
    unscaled_tensor = resized.astype(np.float32)
    return np.expand_dims(unscaled_tensor, axis=0), (x1, y1, x2, y2), []

# ============================================================
# MC DROPOUT PREDICTION ENGINE (+ TTA)
# ============================================================
def predict_age(image_tensor, num_passes=10, use_tta=True):
    age_bins = np.arange(101)
    all_preds = []

    for _ in range(num_passes):
        probs = model(image_tensor, training=False).numpy()[0]
        
        if use_tta:
            flipped_tensor = image_tensor[:, :, ::-1, :]
            probs_flipped = model(flipped_tensor, training=False).numpy()[0]
            probs = (probs + probs_flipped) / 2.0
            
        expected_age = np.sum(age_bins * probs)
        all_preds.append((expected_age, probs))

    ages = [p[0] for p in all_preds]
    mean_probs = np.mean([p[1] for p in all_preds], axis=0)
    return float(np.mean(ages)), float(np.std(ages)), mean_probs

# ============================================================
# GRAPH-NATIVE GRAD-CAM (DEX EXPECTED VALUE GRADIENT)
# ============================================================
def generate_gradcam(image_tensor):
    # 1. Locate inner backbone submodel and the target convolutional layer
    backbone = None
    target_layer = None

    for layer in model.layers:
        if hasattr(layer, 'get_layer'):
            try:
                target_layer = layer.get_layer('top_conv')
                backbone = layer
                break
            except ValueError:
                pass

    if backbone is None or target_layer is None:
        # Fallback to direct top_conv or last Conv2D
        for layer in reversed(model.layers):
            if layer.name == 'top_conv' or isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break

    # 2. Extract feature-maps through a lightweight backbone sub-model
    if backbone is not None:
        backbone_grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[target_layer.output, backbone.output]
        )
        
        # Identify head layers dynamically
        backbone_idx = model.layers.index(backbone)
        head_layers = model.layers[backbone_idx + 1:]
        gap_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
        gmp_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.GlobalMaxPooling2D))
        concat_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.Concatenate))
        dense_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.Dense))

        with tf.GradientTape() as tape:
            # Evaluate feature outputs in eager execution
            conv_outputs, bb_outputs = backbone_grad_model(image_tensor, training=False)
            tape.watch(conv_outputs)

            # Pass eager activations through the classification head
            gap_out = gap_layer(bb_outputs)
            gmp_out = gmp_layer(bb_outputs)
            concat_out = concat_layer([gap_out, gmp_out])
            predictions = dense_layer(concat_out)

            # DEX expected age evaluation
            age_bins = tf.range(101, dtype=tf.float32)
            expected_age = tf.reduce_sum(predictions * age_bins, axis=-1)
    else:
        # Fallback for standard flat models
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor, training=False)
            age_bins = tf.range(101, dtype=tf.float32)
            expected_age = tf.reduce_sum(predictions * age_bins, axis=-1)

    # 3. Compute gradients of expected age w.r.t target layer activations
    grads = tape.gradient(expected_age, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # 4. Generate normalized heatmap matrix
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()    
def overlay_gradcam(original_img_array, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ============================================================
# USER INTERFACE
# ============================================================
st.set_page_config(page_title="Age Estimation AI", page_icon="👤", layout="wide")

st.sidebar.header("⚙️ Model Diagnostics")
st.sidebar.success(f"✅ MC Dropout Active\n(Patched Layers: {patched_layers_count}, Startup Var: {startup_var:.4f})")

st.title("AI Age Estimation System")
st.caption("EfficientNet-B3 DEX | MC Dropout Uncertainty | Grad-CAM Explainability")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with st.spinner("Processing face alignment and MC inference..."):
        tensor, box, warnings = crop_and_align_face(image_pil)

    for w in warnings:
        st.warning(w)

    if tensor is not None:
        mean_age, std_age, mean_probs = predict_age(tensor, num_passes=10, use_tta=True)

        with col1:
            st.image(image_pil, caption="Uploaded Input", use_container_width=True)

        with col2:
            st.metric("Predicted Age", f"{mean_age:.1f} years")
            st.metric("Uncertainty Range (±1σ)", f"{mean_age-std_age:.1f} – {mean_age+std_age:.1f} years (σ = {std_age:.2f})")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(np.arange(101), mean_probs, color='steelblue', label='Output PDF')
            ax.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.1f}')
            ax.fill_between(
                np.arange(101), mean_probs, alpha=0.25, color='steelblue',
                where=(np.arange(101) >= mean_age-std_age) & (np.arange(101) <= mean_age+std_age),
                label='±1σ Bounds'
            )
            ax.set_xlabel("Age Class")
            ax.set_ylabel("Probability")
            ax.legend()
            st.pyplot(fig)

        with st.expander("🔍 View Grad-CAM Visual Explainability"):
            with st.spinner("Generating attention map..."):
                heatmap = generate_gradcam(tensor)
                face_crop_display = tensor[0].astype(np.uint8)
                overlay = overlay_gradcam(face_crop_display, heatmap)
                
            st.image(overlay, caption="Grad-CAM Attention Overlay (Red = Highest Activation)", use_container_width=True)
    else:
        st.error("Face detection failed. Please upload a clear photo with a visible face.")
