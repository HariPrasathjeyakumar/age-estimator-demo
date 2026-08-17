# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import cv2

# ============================================================
# CACHED MODEL LOADING (loads once, not on every interaction)
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_soft_label_model.keras", compile=False)
    return model

@st.cache_resource
def load_detector():
    return MTCNN()

model = load_model()
detector = MTCNN()

# Patch dropout for MC Dropout uncertainty (same as your validated approach)
def patch_dropout_layers(layer_container):
    layers = getattr(layer_container, 'layers', getattr(layer_container, 'submodules', []))
    for layer in layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            if not hasattr(layer, '_mc_patched'):
                orig_call = layer.call
                layer.call = lambda inputs, *args, _orig=orig_call, **kwargs: _orig(inputs, training=True)
                layer._mc_patched = True
        if hasattr(layer, 'layers') or hasattr(layer, 'submodules'):
            patch_dropout_layers(layer)

patch_dropout_layers(model)

# ============================================================
# QUALITY GATING (borrowed idea, cheap addition)
# ============================================================
def check_image_quality(face_crop_array):
    gray = cv2.cvtColor(face_crop_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    warnings = []
    if blur_score < 50:
        warnings.append(f"⚠️ Image appears blurry (score: {blur_score:.1f})")
    if face_crop_array.shape[0] < 80 or face_crop_array.shape[1] < 80:
        warnings.append("⚠️ Face resolution is low — accuracy may be reduced")
    return warnings

# ============================================================
# FACE DETECTION + PREPROCESSING (matches training exactly)
# ============================================================
def crop_and_align_face(image_pil, target_size=(256, 256)):
    img_array = np.array(image_pil.convert('RGB'))
    results = detector.detect_faces(img_array)

    if not results:
        return None, None, ["⚠️ No face detected — cannot generate a reliable prediction"]

    best_face = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)
    pad_x, pad_y = int(w * 0.20), int(h * 0.20)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img_array.shape[1], x + w + pad_x), min(img_array.shape[0], y + h + pad_y)

    face_crop = img_array[y1:y2, x1:x2]
    warnings = check_image_quality(face_crop)

    resized = cv2.resize(face_crop, target_size)
    normalized = resized.astype(np.float32) / 255.0  # matches your validated training normalization
    return np.expand_dims(normalized, axis=0), (x1, y1, x2, y2), warnings

# ============================================================
# MC DROPOUT PREDICTION + TTA
# ============================================================
def predict_age(image_tensor, num_passes=10, use_tta=True):
    age_bins = np.arange(101)
    all_preds = []

    for _ in range(num_passes):
        probs = model(image_tensor, training=False).numpy()[0]
        if use_tta:
            flipped = image_tensor[:, :, ::-1, :]
            probs_flipped = model(flipped, training=False).numpy()[0]
            probs = (probs + probs_flipped) / 2
        expected_age = np.sum(age_bins * probs)
        all_preds.append((expected_age, probs))

    ages = [p[0] for p in all_preds]
    mean_probs = np.mean([p[1] for p in all_preds], axis=0)
    return float(np.mean(ages)), float(np.std(ages)), mean_probs

# ============================================================
# GRAD-CAM
# ============================================================
def generate_gradcam(image_tensor, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_tensor)
        age_bins = tf.range(101, dtype=tf.float32)
        expected_age = tf.reduce_sum(predictions * age_bins, axis=-1)
    grads = tape.gradient(expected_age, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img_array, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Age Estimation AI", page_icon="👤", layout="wide")
st.title("🎯 AI Age Estimation System")
st.caption("EfficientNet-B3 DEX | MC Dropout Uncertainty | Grad-CAM Explainability")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with st.spinner("Analyzing..."):
        tensor, box, warnings = crop_and_align_face(image_pil)

    for w in warnings:
        st.warning(w)

    if tensor is not None:
        mean_age, std_age, mean_probs = predict_age(tensor, num_passes=10, use_tta=True)

        with col1:
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.metric("Predicted Age", f"{mean_age:.1f} years")
            st.metric("Confidence Range (±1σ)", f"{mean_age-std_age:.1f} – {mean_age+std_age:.1f} years")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(np.arange(101), mean_probs, color='steelblue')
            ax.axvline(mean_age, color='red', linestyle='--', label=f'Predicted: {mean_age:.1f}')
            ax.fill_between(np.arange(101), mean_probs, alpha=0.2,
                            where=(np.arange(101) >= mean_age-std_age) & (np.arange(101) <= mean_age+std_age))
            ax.set_xlabel("Age")
            ax.set_ylabel("Probability")
            ax.legend()
            st.pyplot(fig)

        with st.expander("🔍 View Grad-CAM Explainability"):
            heatmap = generate_gradcam(tensor, model)
            face_crop_display = (tensor[0] * 255).astype(np.uint8)
            overlay = overlay_gradcam(face_crop_display, heatmap)
            st.image(overlay, caption="Model attention (red = high influence)", use_column_width=True)
            st.caption("Confirms the model focuses on facial aging features (eyes, forehead, skin texture)")
    else:
        st.error("Could not process this image. Please try a clearer photo with a visible face.")

st.markdown("---")
st.caption("⚠️ This tool provides an AI-generated estimate, not a verified or legal age determination.")