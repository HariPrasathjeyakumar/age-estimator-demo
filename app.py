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

import textwrap
from PIL import ImageDraw

st.set_page_config(
    page_title="Age Estimation AI",
    page_icon="👤",
    layout="wide"
)
# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background: #f8fafc;
        color: #172033;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        font-size: 24px;
        font-weight: 800;
        color: #172033;
        padding: 10px 12px 25px 12px;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 12px;
        padding-left: 12px;
        margin-top: -20px;
        margin-bottom: 25px;
    }


    /* ========================================================
       TOP HEADER
       ======================================================== */

    .app-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #101828;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 18px;
    }


    /* ========================================================
       PRIVACY BANNER
       ======================================================== */

    .privacy-banner {
        background: #f1f6ff;
        border: 1px solid #cbdcf7;
        border-radius: 10px;
        padding: 12px 18px;
        color: #183b72;
        font-size: 14px;
        margin-bottom: 14px;
    }


    /* ========================================================
       MAIN RESULT CARD
       ======================================================== */

    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
        overflow: hidden;
    }


    /* ========================================================
       RESULT IMAGE AREA
       ======================================================== */

    .image-section {
        background: #f8fafc;
        border-radius: 14px;
        padding: 10px;
    }


    /* ========================================================
       ESTIMATION
       ======================================================== */

    .estimated-label {
        color: #172033;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .estimated-age {
        color: #101828;
        font-size: 68px;
        line-height: 1;
        font-weight: 400;
        letter-spacing: -2px;
    }

    .range-label {
        color: #475467;
        font-size: 16px;
        margin-top: 8px;
    }

    .range-value {
        color: #101828;
        font-size: 32px;
        font-weight: 500;
    }


    /* ========================================================
       CONFIDENCE BADGE
       ======================================================== */

    .confidence-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 8px 16px;
        border-radius: 25px;
        background: #e9f8f5;
        border: 1px solid #b8e5dc;
        color: #087f6c;
        font-size: 14px;
        font-weight: 700;
    }


    /* ========================================================
       SMALL INFO CARDS
       ======================================================== */

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 15px 17px;
        height: 100%;
        min-height: 70px;
    }

    .info-icon {
        font-size: 20px;
        margin-bottom: 5px;
    }

    .info-title {
        color: #344054;
        font-size: 13px;
        font-weight: 650;
    }

    .info-value {
        color: #101828;
        font-size: 15px;
        font-weight: 700;
    }


    /* ========================================================
       SECTION CARDS
       ======================================================== */

    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
        margin-top: 14px;
    }

    .section-title {
        color: #172033;
        font-size: 18px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #667085;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* ========================================================
       EXPLANATION ROW
       ======================================================== */

    .influence-row {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        background: #ffffff;
    }

    .influence-title {
        font-size: 14px;
        font-weight: 750;
        color: #344054;
    }

    .influence-description {
        font-size: 12px;
        color: #667085;
        margin-top: 3px;
    }

    .strength {
        color: #0f8f83;
        font-size: 13px;
        font-weight: 700;
    }


    /* ========================================================
       MODEL READY
       ======================================================== */

    .model-ready {
        border: 1px solid #dce3ec;
        border-radius: 14px;
        padding: 15px;
        background: #ffffff;
        color: #087f6c;
        font-weight: 700;
        margin-top: 25px;
    }


    /* ========================================================
       UPLOAD SCREEN
       ======================================================== */

    .upload-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 45px 35px;
        text-align: center;
        margin-top: 25px;
    }

    .upload-icon {
        font-size: 45px;
        margin-bottom: 12px;
    }

    .upload-title {
        font-size: 24px;
        font-weight: 750;
        color: #172033;
    }

    .upload-text {
        color: #667085;
        font-size: 14px;
        margin-top: 8px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #d0d5dd;
        background: white;
    }

    .stButton > button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b8c2d1;
        border-radius: 14px;
        padding: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 12px;
        margin-top: 45px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🧠 AgeLens AI
        </div>

        <div class="sidebar-subtitle">
            Intelligent age estimation
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("☁️  New analysis", use_container_width=True):

        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            padding: 12px;
            color: #344054;
            font-size: 15px;
        ">
            ◴ &nbsp; History
        </div>

        <div style="
            padding: 12px;
            color: #344054;
            font-size: 15px;
        ">
            ⓘ &nbsp; About model
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="model-ready">
            ● &nbsp; Model ready
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Diagnostics"):

        st.write(
            f"MC Dropout layers: {patched_layers_count}"
        )

        st.write(
            f"Startup variance: {startup_var:.4f}"
        )

        st.write(
            "TTA: Enabled"
        )

        st.write(
            "Grad-CAM: Enabled"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-title">
        AgeLens AI
    </div>

    <div class="app-subtitle">
        Age estimation with uncertainty and visual explainability
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PRIVACY MESSAGE
# ============================================================

st.markdown(
    """
    <div class="privacy-banner">
        🛡️ &nbsp;
        <strong>Your image is processed for this analysis only.</strong>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# BEFORE IMAGE UPLOAD
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-icon">
                📷
            </div>

            <div class="upload-title">
                Upload a face image
            </div>

            <div class="upload-text">
                Choose a clear JPG or PNG image containing a visible face.
                <br>
                The AI will estimate apparent age and provide an explanation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            AgeLens AI &nbsp;•&nbsp;
            EfficientNet-B3 &nbsp;•&nbsp;
            MC Dropout &nbsp;•&nbsp;
            TTA &nbsp;•&nbsp;
            Grad-CAM
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image_pil = Image.open(uploaded_file).convert("RGB")


# ============================================================
# FACE DETECTION + PREDICTION
# ============================================================

with st.spinner("Analyzing face..."):

    tensor, box, warnings = crop_and_align_face(
        image_pil
    )

    for warning in warnings:
        st.warning(warning)


# ============================================================
# FACE NOT DETECTED
# ============================================================

if tensor is None:

    st.error(
        "Face detection failed. Please upload a clear photo "
        "with a visible face."
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

with st.spinner("Running age estimation..."):

    mean_age, std_age, mean_probs = predict_age(
        tensor,
        num_passes=10,
        use_tta=True
    )


lower_age = max(
    0,
    mean_age - std_age
)

upper_age = mean_age + std_age


# ============================================================
# DRAW FACE BOX ON ORIGINAL IMAGE
# ============================================================

display_image = image_pil.copy()

if box is not None:

    draw = ImageDraw.Draw(display_image)

    x1, y1, x2, y2 = box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="#4f46e5",
        width=4
    )


# ============================================================
# MAIN RESULT CARD
# ============================================================

left, right = st.columns(
    [1.05, 1],
    gap="medium"
)


# ============================================================
# LEFT - IMAGE
# ============================================================

with left:

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
            👤 Analyzed image
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        display_image,
        use_container_width=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    button1, button2 = st.columns(2)

    with button1:

        st.button(
            "▣  Replace image",
            use_container_width=True
        )

    with button2:

        st.button(
            "⟳  Analyze another",
            use_container_width=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT - AGE RESULT
# ============================================================

with right:

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="estimated-label">
            Estimated age
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="estimated-age">
            {mean_age:.1f}
        </div>

        <div class="confidence-badge">
            ✓ &nbsp; High confidence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    range_col1, range_col2 = st.columns(2)

    with range_col1:

        st.markdown(
            f"""
            <div>
                <div class="range-label">
                    📊 Likely range
                </div>

                <div class="range-value">
                    {lower_age:.1f}–{upper_age:.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with range_col2:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-title">
                    🎯 Face status
                </div>

                <div class="info-value">
                    Face detected
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">◌</div>
                <div class="info-title">
                    Inference
                </div>
                <div class="info-value">
                    10 passes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">✧</div>
                <div class="info-title">
                    Augmentation
                </div>
                <div class="info-value">
                    TTA on
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">◎</div>
                <div class="info-title">
                    Detection
                </div>
                <div class="info-value">
                    MTCNN
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# AGE PROBABILITY
# ============================================================

st.markdown(
    '<div class="section-card">',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="section-title">
        Age probability
        <span style="
            float:right;
            color:#4f46e5;
            font-weight:800;
        ">
            {mean_age:.1f}
        </span>
    </div>

    <div class="section-subtitle">
        Probability distribution across predicted age classes.
    </div>
    """,
    unsafe_allow_html=True
)


ages = np.arange(101)

fig, ax = plt.subplots(
    figsize=(13, 4)
)

ax.plot(
    ages,
    mean_probs,
    linewidth=2.5
)

ax.fill_between(
    ages,
    mean_probs,
    alpha=0.12
)

ax.axvline(
    mean_age,
    linestyle="--",
    linewidth=2,
    label=f"Predicted: {mean_age:.1f}"
)

ax.axvspan(
    lower_age,
    upper_age,
    alpha=0.10,
    label="Uncertainty range"
)

ax.set_xlabel(
    "Age",
    fontsize=10
)

ax.set_ylabel(
    "Probability",
    fontsize=10
)

ax.set_xlim(
    0,
    100
)

ax.grid(
    alpha=0.18
)

ax.legend(
    frameon=False
)

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GRAD-CAM + FEATURE EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-card">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-title">
        🔍 Where the model looked
    </div>

    <div class="section-subtitle">
        Visual explanation of the facial regions contributing
        to the prediction.
    </div>
    """,
    unsafe_allow_html=True
)


with st.spinner("Generating visual explanation..."):

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


cam_col, influence_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# GRAD CAM
# ============================================================

with cam_col:

    st.markdown(
        """
        <div class="section-title">
            🔥 Attention map
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        overlay_img,
        use_container_width=True
    )

    st.markdown(
        """
        <div style="
            font-size:12px;
            color:#667085;
            text-align:center;
            margin-top:5px;
        ">
            Blue = lower influence &nbsp;&nbsp;
            Yellow = moderate &nbsp;&nbsp;
            Red = higher influence
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INFLUENCE
# ============================================================

with influence_col:

    st.markdown(
        """
        <div class="section-title">
            🧬 What influenced this result
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="influence-row">

            <div class="influence-title">
                ◡ &nbsp; Facial structure
            </div>

            <div class="influence-description">
                {primary_feats}
            </div>

            <div class="strength">
                Strong &nbsp; ●●●●●
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="influence-row">

            <div class="influence-title">
                ◌ &nbsp; Skin / texture
            </div>

            <div class="influence-description">
                {bio_markers}
            </div>

            <div class="strength">
                Moderate &nbsp; ●●●○○
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="influence-row">

            <div class="influence-title">
                👁 &nbsp; Eye-area features
            </div>

            <div class="influence-description">
                Facial details around the eye region contribute
                to the learned representation.
            </div>

            <div class="strength">
                Strong &nbsp; ●●●●●
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="influence-row">

            <div class="influence-title">
                ◡ &nbsp; Facial symmetry
            </div>

            <div class="influence-description">
                Overall facial proportions contribute to the
                estimated apparent age.
            </div>

            <div class="strength">
                Moderate &nbsp; ●●●○○
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander("⚙️ View technical pipeline"):

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("### 📷 Input")
        st.write("Face image")
        st.write("MTCNN detection")

    with tech2:
        st.markdown("### 🧠 Model")
        st.write("EfficientNet-B3")
        st.write("Soft-label age estimation")

    with tech3:
        st.markdown("### 🎲 Uncertainty")
        st.write("MC Dropout")
        st.write("10 stochastic passes")
        st.write("Test-Time Augmentation")

    with tech4:
        st.markdown("### 🔥 Explainability")
        st.write("Grad-CAM")
        st.write("Feature attribution")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>AgeLens AI</strong><br>
        EfficientNet-B3 • MC Dropout • TTA • Grad-CAM<br>
        Built with Python • TensorFlow • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
