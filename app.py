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
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       MAIN APPLICATION
       ============================== */

    .stApp {
        background-color: #f6f8fc;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ==============================
       HEADER
       ============================== */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 50%,
            #312e81 100%
        );

        padding: 35px 40px;
        border-radius: 22px;
        margin-bottom: 30px;

        box-shadow:
            0 12px 35px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 8px;
    }

    .hero-status {
        display: inline-block;
        margin-top: 18px;
        padding: 6px 14px;
        border-radius: 20px;

        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.35);

        color: #86efac;
        font-size: 13px;
        font-weight: 600;
    }


    /* ==============================
       SECTION TITLES
       ============================== */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ==============================
       CARDS
       ============================== */

    div[data-testid="stMetric"] {

        background: white;

        border: 1px solid #e2e8f0;

        border-radius: 18px;

        padding: 22px;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.06);

        min-height: 125px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800;
    }


    /* ==============================
       FILE UPLOADER
       ============================== */

    div[data-testid="stFileUploader"] {

        background: white;

        border: 2px dashed #cbd5e1;

        border-radius: 18px;

        padding: 20px;

        transition: 0.2s;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: #fafaff;
    }


    /* ==============================
       IMAGE
       ============================== */

    img {
        border-radius: 15px;
    }


    /* ==============================
       INFO BOX
       ============================== */

    .info-box {

        background: white;

        border: 1px solid #e2e8f0;

        border-radius: 18px;

        padding: 24px;

        box-shadow:
            0 5px 20px rgba(15,23,42,0.05);

        margin-bottom: 20px;
    }

    .info-heading {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 10px;
    }

    .info-text {
        color: #475569;
        font-size: 14px;
        line-height: 1.7;
    }


    /* ==============================
       PIPELINE CARDS
       ============================== */

    .pipeline {

        background: white;

        border: 1px solid #e2e8f0;

        border-radius: 16px;

        padding: 20px;

        text-align: center;

        min-height: 130px;

        box-shadow:
            0 5px 18px rgba(15,23,42,0.04);
    }

    .pipeline-number {
        color: #6366f1;
        font-size: 13px;
        font-weight: 700;
    }

    .pipeline-title {
        color: #111827;
        font-size: 16px;
        font-weight: 700;
        margin-top: 8px;
    }

    .pipeline-text {
        color: #64748b;
        font-size: 12px;
        margin-top: 6px;
    }


    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Information")

    st.success("Model Ready")

    st.markdown("---")

    st.markdown("### Architecture")

    st.write("**EfficientNet-B3 DEX**")

    st.markdown("### Face Detection")

    st.write("**MTCNN**")

    st.markdown("### Uncertainty")

    st.write("**MC Dropout**")

    st.markdown("### Explainability")

    st.write("**Grad-CAM**")

    st.markdown("### Test-Time Augmentation")

    st.write("**Enabled**")

    st.markdown("---")

    st.caption("Advanced Diagnostics")

    st.write(
        f"Patched Dropout Layers: "
        f"**{patched_layers_count}**"
    )

    st.write(
        f"Startup Variance: "
        f"**{startup_var:.4f}**"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 AI Age Estimation
        </div>

        <div class="hero-subtitle">
            Explainable and uncertainty-aware facial age prediction
            using deep learning and computer vision.
        </div>

        <div class="hero-status">
            ● MODEL READY
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Face Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a clear face image to begin age estimation.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a face image above to start the analysis."
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🔬 AI Analysis Pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Our system processes the image through multiple '
        'computer vision and deep learning stages.'
        '</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            """
            <div class="pipeline">
                <div class="pipeline-number">STEP 01</div>
                <div class="pipeline-title">📷 Input</div>
                <div class="pipeline-text">
                    Upload a face image
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            """
            <div class="pipeline">
                <div class="pipeline-number">STEP 02</div>
                <div class="pipeline-title">👤 Detection</div>
                <div class="pipeline-text">
                    MTCNN detects and aligns the face
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:
        st.markdown(
            """
            <div class="pipeline">
                <div class="pipeline-number">STEP 03</div>
                <div class="pipeline-title">🧠 Inference</div>
                <div class="pipeline-text">
                    EfficientNet-B3 predicts age
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p4:
        st.markdown(
            """
            <div class="pipeline">
                <div class="pipeline-number">STEP 04</div>
                <div class="pipeline-title">🔍 Explanation</div>
                <div class="pipeline-text">
                    Grad-CAM explains the prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file:

    image_pil = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    with st.spinner(
        "🔄 Detecting face and preparing image..."
    ):

        tensor, box, warnings = crop_and_align_face(
            image_pil
        )


    for warning in warnings:
        st.warning(warning)


    # ========================================================
    # SUCCESS
    # ========================================================

    if tensor is not None:

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Running AI age estimation..."
        ):

            mean_age, std_age, mean_probs = predict_age(
                tensor,
                num_passes=10,
                use_tta=True
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if std_age <= 2:
            confidence = "HIGH"
        elif std_age <= 5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"


        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🎯 Analysis Result'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'AI-generated age prediction and uncertainty analysis.'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # METRICS
        # ====================================================

        m1, m2, m3 = st.columns(3)


        with m1:

            st.metric(
                "PREDICTED AGE",
                f"{mean_age:.1f} years"
            )


        with m2:

            st.metric(
                "CONFIDENCE",
                confidence
            )


        with m3:

            st.metric(
                "UNCERTAINTY",
                f"± {std_age:.2f} years"
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
                '<div class="section-title">'
                '👤 Uploaded Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_pil,
                use_container_width=True
            )

            if box is not None:

                st.success(
                    "Face detected successfully"
                )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        with summary_col:

            st.markdown(
                '<div class="section-title">'
                '📋 Prediction Summary'
                '</div>',
                unsafe_allow_html=True
            )

            lower_age = max(
                0,
                mean_age - std_age
            )

            upper_age = (
                mean_age + std_age
            )

            st.markdown(
                f"""
                <div class="info-box">

                    <div class="info-heading">
                        Age Estimation
                    </div>

                    <div class="info-text">

                        The model estimates the apparent
                        age of the detected face as

                        <strong>
                        {mean_age:.1f} years
                        </strong>.

                        <br><br>

                        Estimated uncertainty range:

                        <strong>
                        {lower_age:.1f} – {upper_age:.1f} years
                        </strong>

                        <br><br>

                        The prediction is generated using
                        multiple stochastic inference passes
                        to estimate model uncertainty.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "💡 Prediction accuracy can be affected by "
                "lighting, face angle, image quality and "
                "facial visibility."
            )


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📊 Age Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Probability assigned by the model to each age class.'
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
            linewidth=2.5
        )


        ax.axvline(
            mean_age,
            linestyle="--",
            linewidth=2,
            label=f"Predicted: {mean_age:.1f}"
        )


        ax.fill_between(
            age_range,
            mean_probs,
            alpha=0.18,
            where=(
                (age_range >= mean_age - std_age)
                &
                (age_range <= mean_age + std_age)
            ),
            label="±1σ uncertainty"
        )


        ax.set_xlabel(
            "Age (years)"
        )

        ax.set_ylabel(
            "Probability"
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
        # GRAD-CAM
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🔍 Explainable AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Understand which facial regions influenced the model prediction.'
            '</div>',
            unsafe_allow_html=True
        )


        with st.spinner(
            "🔥 Generating Grad-CAM explanation..."
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
        # GRAD-CAM IMAGE
        # ----------------------------------------------------

        with cam_col:

            st.markdown(
                '<div class="section-title">'
                '🔥 Attention Map'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                overlay_img,
                use_container_width=True
            )

            st.caption(
                "Warm regions represent stronger model activation."
            )


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        with explanation_col:

            st.markdown(
                '<div class="section-title">'
                '🧬 Model Interpretation'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="info-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                "**🎯 Dominant Facial Cues**"
            )

            st.write(
                primary_feats
            )

            st.markdown(
                "**🧬 Biological Indicators**"
            )

            st.write(
                bio_markers
            )

            st.markdown(
                "**💡 Model Reasoning**"
            )

            st.write(
                f"""
                For an estimated age of **{mean_age:.1f} years**,
                the neural network gives greater importance to
                facial regions associated with facial geometry
                and dermal texture.
                """
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # TECHNICAL PIPELINE
        # ====================================================

        st.markdown("---")

        with st.expander(
            "⚙️ View Technical Pipeline"
        ):

            t1, t2, t3, t4 = st.columns(4)


            with t1:

                st.markdown(
                    """
                    **01 — INPUT**

                    📷

                    Upload face image
                    """
                )


            with t2:

                st.markdown(
                    """
                    **02 — DETECTION**

                    👤

                    MTCNN face detection
                    """
                )


            with t3:

                st.markdown(
                    """
                    **03 — INFERENCE**

                    🧠

                    EfficientNet-B3  
                    MC Dropout + TTA
                    """
                )


            with t4:

                st.markdown(
                    """
                    **04 — EXPLANATION**

                    🔍

                    Grad-CAM
                    """
                )


        # ====================================================
        # FOOTER
        # ====================================================

        st.markdown(
            """
            <div class="footer">

                AI Age Estimation System

                <br>

                EfficientNet-B3 • MC Dropout • TTA • Grad-CAM

                <br>

                Built with Python • TensorFlow • Streamlit

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FACE DETECTION FAILURE
    # ========================================================

    else:

        st.error(
            "❌ Face detection failed. "
            "Please upload a clear image with a visible face."
        )
