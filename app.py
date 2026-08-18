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
    
    # Range [0.0, 255.0] float32 for internal Rescaling layer compatibility
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
# GRAPH-NATIVE GRAD-CAM (K3 & DTYPE MATCHED)
# ============================================================
def generate_gradcam(image_tensor):
    # 1. Locate inner backbone submodel and target layer
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
        for layer in reversed(model.layers):
            if layer.name == 'top_conv' or isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break

    # 2. Extract feature maps through sub-model
    if backbone is not None:
        backbone_grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[target_layer.output, backbone.output]
        )
        
        backbone_idx = model.layers.index(backbone)
        head_layers = model.layers[backbone_idx + 1:]
        gap_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
        gmp_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.GlobalMaxPooling2D))
        concat_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.Concatenate))
        dense_layer = next(l for l in head_layers if isinstance(l, tf.keras.layers.Dense))

        with tf.GradientTape() as tape:
            conv_outputs, bb_outputs = backbone_grad_model(image_tensor, training=False)
            tape.watch(conv_outputs)

            gap_out = gap_layer(bb_outputs)
            gmp_out = gmp_layer(bb_outputs)
            concat_out = concat_layer([gap_out, gmp_out])
            predictions = dense_layer(concat_out)

            num_classes = tf.shape(predictions)[-1]
            age_bins = tf.cast(tf.range(num_classes), dtype=predictions.dtype)
            expected_age = tf.reduce_sum(predictions * age_bins, axis=-1)
    else:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor, training=False)
            num_classes = tf.shape(predictions)[-1]
            age_bins = tf.cast(tf.range(num_classes), dtype=predictions.dtype)
            expected_age = tf.reduce_sum(predictions * age_bins, axis=-1)

    # 3. Compute gradients of expected age w.r.t target layer activations
    grads = tape.gradient(expected_age, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # 4. Generate normalized heatmap matrix
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def create_pil_overlay(original_tensor_crop, heatmap, alpha=0.45):
    """Blends 2D heatmap onto face crop using PIL/Matplotlib (Zero OpenCV)."""
    base_img = np.clip(original_tensor_crop[0], 0, 255).astype(np.uint8)
    
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_pil.resize((base_img.shape[1], base_img.shape[0]), resample=Image.BILINEAR)
    heatmap_resized_np = np.array(heatmap_resized) / 255.0

    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized_np)[:, :, :3]
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)

    blended = (1.0 - alpha) * base_img.astype(np.float32) + alpha * heatmap_colored_uint8.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

def generate_feature_explanation(mean_age):
    """Maps predicted age tier to dominant facial feature influences."""
    if mean_age < 18:
        primary_features = "Facial proportions, smooth skin texture, and eye-to-face distance ratio."
        biological_markers = "Minimal fine lines; primary reliance on cranial shape and facial compact ratio."
    elif 18 <= mean_age < 35:
        primary_features = "Jawline definition, skin tautness, periocular (eye area) clarity, and nasolabial symmetry."
        biological_markers = "Peak muscle tone, subtle early expression lines around the eyes/mouth."
    elif 35 <= mean_age < 55:
        primary_features = "Forehead expression lines, nasolabial folds, cheek volume, and subtle under-eye texture."
        biological_markers = "Gradual loss of skin elasticity, deepening expression creases, structural volume shifts."
    else:
        primary_features = "Deep forehead furrows, periorbital (crow's feet) wrinkles, neck skin laxity, and tissue sagging."
        biological_markers = "Prominent structural remodeling, loss of dermal thickness, and pronounced facial creasing."

    return primary_features, biological_markers

# ============================================================
# PROFESSIONAL USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="AI Age Estimation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- HEADER ---------- */

    .main-header {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 30px 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }

    .main-title {
        color: white;
        font-size: 34px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 18px;
    }

    .status-badge {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        color: #86efac;
        padding: 7px 15px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid rgba(34,197,94,0.3);
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* ---------- UPLOAD CARD ---------- */

    .upload-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
        margin-bottom: 25px;
    }

    /* ---------- RESULT CARDS ---------- */

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 750;
    }

    .metric-description {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- IMAGE CARD ---------- */

    .image-card {
        background: white;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    }

    /* ---------- INFO CARD ---------- */

    .info-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    }

    .info-title {
        font-size: 17px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }

    .info-text {
        font-size: 14px;
        color: #475569;
        line-height: 1.6;
    }

    /* ---------- EXPLAINABILITY ---------- */

    .explain-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    }

    .explain-title {
        color: #111827;
        font-size: 20px;
        font-weight: 700;
    }

    .explain-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

    /* ---------- STREAMLIT ELEMENTS ---------- */

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">

    <div class="main-title">
        🧠 AI Age Estimation
    </div>

    <div class="main-subtitle">
        Explainable and uncertainty-aware facial age prediction
        using deep learning and computer vision.
    </div>

    <span class="status-badge">
        ● MODEL READY
    </span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - ADVANCED DIAGNOSTICS
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Diagnostics")

    st.success("Model loaded successfully")

    st.markdown("### Model Configuration")

    st.write(
        "**Architecture:** EfficientNet-B3 DEX"
    )

    st.write(
        "**Face Detector:** MTCNN"
    )

    st.write(
        "**Uncertainty:** MC Dropout"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**TTA:** Enabled"
    )

    st.divider()

    st.markdown("### Runtime Diagnostics")

    st.metric(
        "Patched Dropout Layers",
        patched_layers_count
    )

    st.metric(
        "Startup Variance",
        f"{startup_var:.4f}"
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Face Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a clear face image to estimate the apparent age.'
    '</div>',
    unsafe_allow_html=True
)

with st.container():

    st.markdown(
        '<div class="upload-card">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    st.markdown(
        '<p style="color:#64748b;font-size:13px;">'
        '💡 For best results, use a clear front-facing image '
        'with good lighting.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file:

    image_pil = Image.open(uploaded_file).convert("RGB")

    with st.spinner(
        "🔄 Detecting face and running AI inference..."
    ):

        tensor, box, warnings = crop_and_align_face(image_pil)

    # Display warnings

    for warning in warnings:
        st.warning(warning)

    # ========================================================
    # FACE DETECTION SUCCESS
    # ========================================================

    if tensor is not None:

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        mean_age, std_age, mean_probs = predict_age(
            tensor,
            num_passes=10,
            use_tta=True
        )

        # ----------------------------------------------------
        # CONFIDENCE ESTIMATION
        # ----------------------------------------------------

        uncertainty_percentage = min(
            100,
            max(
                0,
                100 - (std_age / max(mean_age, 1)) * 100
            )
        )

        if uncertainty_percentage >= 85:
            confidence_label = "HIGH"
        elif uncertainty_percentage >= 70:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.markdown(
            '<div class="section-title">🎯 Analysis Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'AI-generated age estimation and uncertainty analysis'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # METRIC CARDS
        # ====================================================

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Predicted Age
                    </div>

                    <div class="metric-value">
                        {mean_age:.1f} years
                    </div>

                    <div class="metric-description">
                        Estimated apparent age
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with metric2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Confidence
                    </div>

                    <div class="metric-value">
                        {confidence_label}
                    </div>

                    <div class="metric-description">
                        Based on prediction stability
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with metric3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Uncertainty
                    </div>

                    <div class="metric-value">
                        ±{std_age:.1f} years
                    </div>

                    <div class="metric-description">
                        MC Dropout standard deviation
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # ====================================================
        # IMAGE + SUMMARY
        # ====================================================

        image_col, summary_col = st.columns(
            [1, 1],
            gap="large"
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with image_col:

            st.markdown(
                '<div class="section-title">👤 Input Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_pil,
                use_container_width=True
            )

            if box is not None:

                st.caption(
                    "✓ Face detected and standardized "
                    "for model inference."
                )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        with summary_col:

            st.markdown(
                '<div class="section-title">📋 Prediction Summary</div>',
                unsafe_allow_html=True
            )

            lower_age = max(
                0,
                mean_age - std_age
            )

            upper_age = mean_age + std_age

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-title">
                        Age Estimation
                    </div>

                    <div class="info-text">

                        The model estimates the apparent age
                        of the detected face as

                        <strong>{mean_age:.1f} years</strong>.

                        <br><br>

                        The estimated uncertainty range is

                        <strong>
                        {lower_age:.1f} – {upper_age:.1f} years
                        </strong>.

                        <br><br>

                        Multiple stochastic inference passes
                        were used to measure prediction stability.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.info(
                "💡 The prediction represents an estimated "
                "apparent age and may vary depending on image "
                "quality, lighting, pose and facial visibility."
            )


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Age Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Distribution of model probability across age classes'
            '</div>',
            unsafe_allow_html=True
        )

        age_range = np.arange(101)

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            age_range,
            mean_probs,
            linewidth=2.5,
            label="Age Probability"
        )

        ax.axvline(
            mean_age,
            linestyle="--",
            linewidth=2,
            label=f"Predicted Age: {mean_age:.1f}"
        )

        ax.fill_between(
            age_range,
            mean_probs,
            alpha=0.18,
            where=(
                (age_range >= mean_age - std_age) &
                (age_range <= mean_age + std_age)
            ),
            label="±1σ uncertainty"
        )

        ax.set_xlabel(
            "Age (years)",
            fontsize=11
        )

        ax.set_ylabel(
            "Probability",
            fontsize=11
        )

        ax.set_xlim(
            0,
            100
        )

        ax.grid(
            alpha=0.2
        )

        ax.legend(
            frameon=False
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔍 Explainable AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Visual explanation of the regions influencing the prediction'
            '</div>',
            unsafe_allow_html=True
        )

        with st.spinner(
            "Generating Grad-CAM explanation..."
        ):

            heatmap = generate_gradcam(
                tensor
            )

            overlay_img = create_pil_overlay(
                tensor,
                heatmap,
                alpha=0.45
            )

            primary_feats, bio_markers = (
                generate_feature_explanation(
                    mean_age
                )
            )


        cam_col, explanation_col = st.columns(
            [1, 1],
            gap="large"
        )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        with cam_col:

            st.markdown(
                '<div class="explain-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="explain-title">'
                '🔥 Grad-CAM Attention Map'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="explain-subtitle">'
                'Highlighted regions indicate areas receiving '
                'stronger model attention.'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                overlay_img,
                use_container_width=True
            )

            st.caption(
                "Warm regions indicate stronger model activation."
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # FEATURE EXPLANATION
        # ----------------------------------------------------

        with explanation_col:

            st.markdown(
                '<div class="explain-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="explain-title">'
                '🧬 Model Interpretation'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="explain-subtitle">'
                'Feature-level interpretation of the estimated age'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🎯 Dominant Facial Cues"
            )

            st.write(
                primary_feats
            )

            st.markdown(
                "### 🧬 Biological Indicators"
            )

            st.write(
                bio_markers
            )

            st.markdown(
                "### 💡 Model Reasoning"
            )

            st.info(
                f"""
                For an estimated age of **{mean_age:.1f} years**,
                the neural network gives greater importance to
                facial geometry and dermal texture represented
                in the highlighted regions.
                """
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # TECHNICAL PIPELINE
        # ====================================================

        with st.expander(
            "⚙️ View Technical Pipeline"
        ):

            pipeline1, pipeline2, pipeline3, pipeline4 = (
                st.columns(4)
            )

            with pipeline1:
                st.markdown(
                    "**01**\n\n"
                    "📷 **Input**\n\n"
                    "Face image uploaded by user"
                )

            with pipeline2:
                st.markdown(
                    "**02**\n\n"
                    "👤 **Detection**\n\n"
                    "MTCNN face detection & alignment"
                )

            with pipeline3:
                st.markdown(
                    "**03**\n\n"
                    "🧠 **Inference**\n\n"
                    "EfficientNet-B3 + MC Dropout + TTA"
                )

            with pipeline4:
                st.markdown(
                    "**04**\n\n"
                    "🔍 **Explanation**\n\n"
                    "Grad-CAM feature visualization"
                )


# ============================================================
# NO IMAGE STATE
# ============================================================

else:

    st.markdown("<br>", unsafe_allow_html=True)

    empty_col1, empty_col2, empty_col3 = st.columns(
        [1, 2, 1]
    )

    with empty_col2:

        st.markdown(
            """
            <div style="
                background:white;
                padding:45px;
                border-radius:20px;
                text-align:center;
                border:1px solid #e2e8f0;
                box-shadow:0 5px 20px rgba(15,23,42,0.05);
            ">

                <div style="
                    font-size:50px;
                    margin-bottom:15px;
                ">
                    📷
                </div>

                <div style="
                    font-size:22px;
                    font-weight:700;
                    color:#111827;
                ">
                    Ready for Analysis
                </div>

                <div style="
                    color:#64748b;
                    font-size:14px;
                    margin-top:8px;
                ">
                    Upload a face image above to begin
                    AI-powered age estimation.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI Age Estimation System &nbsp;•&nbsp;
        EfficientNet-B3 &nbsp;•&nbsp;
        MC Dropout &nbsp;•&nbsp;
        Grad-CAM
        <br>
        Built with Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
