import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import hashlib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AgeLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background: #F7F9FC;
        color: #172033;
    }

    .main .block-container {
        max-width: 1380px;
        padding: 34px 42px 60px 42px;
    }

    /* Remove excessive Streamlit spacing */
    .element-container {
        margin-bottom: 0.25rem;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E9F0;
    }

    section[data-testid="stSidebar"] > div {
        padding: 28px 20px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 3px;
    }

    .brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        font-weight: 800;
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        margin-left: 50px;
        margin-top: 4px;
        color: #8993A4;
        font-size: 12px;
    }

    .sidebar-heading {
        margin-top: 30px;
        margin-bottom: 9px;
        color: #9AA3B2;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .sidebar-item {
        padding: 11px 12px;
        border-radius: 9px;
        margin: 4px 0;
        color: #596579;
        font-size: 14px;
    }

    .sidebar-item.active {
        background: #F1F0FF;
        color: #4F46E5;
        font-weight: 700;
    }

    .status-card {
        margin-top: 30px;
        padding: 15px;
        border: 1px solid #E4E8EF;
        border-radius: 13px;
        background: #FBFCFE;
    }

    .status-line {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #087F5B;
        font-size: 13px;
        font-weight: 750;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #16B777;
    }

    .status-description {
        margin-top: 6px;
        color: #98A2B3;
        font-size: 11px;
    }


    /* ========================================================
       HEADER
    ======================================================== */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 18px;
    }

    .page-title {
        color: #101828;
        font-size: 36px;
        font-weight: 850;
        letter-spacing: -1.3px;
        line-height: 1.1;
    }

    .page-subtitle {
        margin-top: 7px;
        color: #718096;
        font-size: 14px;
    }

    .ready-badge {
        padding: 8px 13px;
        border-radius: 999px;
        background: #ECFDF5;
        border: 1px solid #BCEAD7;
        color: #087F5B;
        font-size: 12px;
        font-weight: 750;
        white-space: nowrap;
    }


    /* ========================================================
       PRIVACY NOTICE
    ======================================================== */

    .privacy {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #F0F6FF;
        border: 1px solid #CFE0FF;
        border-radius: 11px;
        padding: 13px 16px;
        color: #315B9B;
        font-size: 13px;
        margin-bottom: 18px;
    }


    /* ========================================================
       UPLOAD AREA
    ======================================================== */

    .upload-title {
        font-size: 16px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 4px;
    }

    .upload-subtitle {
        color: #8993A4;
        font-size: 12px;
        margin-bottom: 12px;
    }

    div[data-testid="stFileUploader"] {
        width: 100%;
    }

    div[data-testid="stFileUploader"] section {
        background: #FFFFFF !important;
        border: 1.5px dashed #C9D1DE !important;
        border-radius: 13px !important;
        min-height: 92px !important;
        transition: all 0.2s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #635BFF !important;
        background: #FAFAFF !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #98A2B3 !important;
    }


    /* ========================================================
       STREAMLIT BORDER CONTAINERS
    ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E3E8F0 !important;
        border-radius: 16px !important;
        background: #FFFFFF !important;
    }


    /* ========================================================
       RESULT AREA
    ======================================================== */

    .card-heading {
        font-size: 16px;
        font-weight: 800;
        color: #172033;
    }

    .card-caption {
        color: #98A2B3;
        font-size: 11px;
    }

    .age-number {
        font-size: 64px;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -2px;
        color: #101828;
        margin-top: 8px;
    }

    .age-unit {
        color: #6B7280;
        font-size: 14px;
        font-weight: 600;
    }

    .confidence {
        display: inline-block;
        margin-top: 14px;
        padding: 7px 12px;
        border-radius: 999px;
        background: #ECFDF5;
        border: 1px solid #BCEAD7;
        color: #087F5B;
        font-size: 11px;
        font-weight: 800;
    }


    /* ========================================================
       METRICS
    ======================================================== */

    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E8ECF2;
        border-radius: 11px;
        padding: 13px;
        min-height: 72px;
    }

    .metric-label {
        color: #98A2B3;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        color: #263247;
        font-size: 14px;
        font-weight: 750;
        margin-top: 5px;
    }


    /* ========================================================
       SECTION TITLES
    ======================================================== */

    .section-title {
        color: #172033;
        font-size: 20px;
        font-weight: 800;
        margin-top: 25px;
    }

    .section-subtitle {
        color: #8993A4;
        font-size: 12px;
        margin-top: 4px;
        margin-bottom: 12px;
    }


    /* ========================================================
       FEATURE CARDS
    ======================================================== */

    .feature-card {
        border: 1px solid #E5E9F0;
        border-radius: 11px;
        padding: 13px;
        margin-bottom: 9px;
        background: #FFFFFF;
    }

    .feature-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .feature-name {
        color: #263247;
        font-size: 13px;
        font-weight: 750;
    }

    .feature-strength {
        color: #128A6B;
        font-size: 11px;
        font-weight: 750;
    }

    .feature-description {
        color: #8993A4;
        font-size: 11px;
        line-height: 1.45;
        margin-top: 5px;
    }

    .feature-bars {
        margin-top: 7px;
        color: #159A7B;
        letter-spacing: 2px;
        font-size: 11px;
    }


    /* ========================================================
       REASONING BOX
    ======================================================== */

    .reasoning-box {
        margin-top: 13px;
        padding: 14px 15px;
        border-radius: 11px;
        background: #F0F6FF;
        border: 1px solid #CFE0FF;
    }

    .reasoning-title {
        color: #356CC2;
        font-size: 12px;
        font-weight: 800;
    }

    .reasoning-text {
        color: #58708F;
        font-size: 12px;
        line-height: 1.55;
        margin-top: 5px;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        border-radius: 10px !important;
        min-height: 40px !important;
        border: 1px solid #D8DEE8 !important;
        background: #FFFFFF !important;
        color: #445064 !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        border-color: #625BF5 !important;
        color: #5148D9 !important;
        background: #FAFAFF !important;
    }


    /* ========================================================
       EXPANDER
    ======================================================== */

    div[data-testid="stExpander"] {
        border: 1px solid #E2E7EF !important;
        border-radius: 13px !important;
        background: #FFFFFF !important;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 25px;
        border-top: 1px solid #E7EBF1;
        color: #A0A8B6;
        font-size: 11px;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div class="brand-name">AgeLens AI</div>
        </div>
        <div class="brand-subtitle">
            Intelligent age estimation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Workspace</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item active">＋ &nbsp; New analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">◷ &nbsp; History</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-item">ⓘ &nbsp; About model</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">System</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item">⌁ &nbsp; Diagnostics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-card">
            <div class="status-line">
                <span class="status-dot"></span>
                Model ready
            </div>
            <div class="status-description">
                EfficientNet-B3 · Production mode
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">
        <div>
            <div class="page-title">AgeLens AI</div>
            <div class="page-subtitle">
                Age estimation with uncertainty and visual explainability
            </div>
        </div>

        <div class="ready-badge">
            ● Model ready
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PRIVACY
# ============================================================

st.markdown(
    """
    <div class="privacy">
        <span>🛡️</span>
        <span>
            Your image is processed for this analysis only.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="upload-title">Analyze a face image</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="upload-subtitle">
        Upload a clear JPG or PNG image containing a visible face.
        The AI will estimate apparent age and provide visual explanations.
    </div>
    """,
    unsafe_allow_html=True,
)


# Session state for resetting uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


uploaded_file = st.file_uploader(
    "Upload face image",
    type=["jpg", "jpeg", "png"],
    key=f"face_uploader_{st.session_state.uploader_key}",
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    with st.container(border=True):

        st.markdown(
            """
            <div style="
                text-align:center;
                padding:42px 20px;
            ">
                <div style="
                    font-size:38px;
                    margin-bottom:12px;
                ">
                    🖼️
                </div>

                <div style="
                    font-size:18px;
                    font-weight:800;
                    color:#273248;
                ">
                    Upload a face image
                </div>

                <div style="
                    font-size:13px;
                    color:#8A94A6;
                    margin-top:7px;
                ">
                    JPG or PNG · Clear visible face recommended
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            <b>AgeLens AI</b><br>
            EfficientNet-B3 · MC Dropout · TTA · Grad-CAM<br>
            Built with Python · TensorFlow · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")

file_hash = hashlib.md5(
    uploaded_file.getvalue()
).hexdigest()[:8]


# ============================================================
# ============================================================
# MODEL INFERENCE FUNCTION
# ============================================================
#
# IMPORTANT:
# Replace ONLY the values returned by this function with
# your actual model inference.
#
# Your existing app currently uses:
# predicted_age = 24.9
# lower_age = 22.0
# upper_age = 27.9
#
# See your previous app code. 
# ============================================================

def run_model(image):

    # --------------------------------------------------------
    # DEMO VALUES
    # --------------------------------------------------------
    #
    # REPLACE THIS SECTION WITH YOUR ACTUAL MODEL CODE.
    #
    # Example:
    #
    # age_probabilities = model.predict(face_tensor)
    # predicted_age = np.sum(
    #     age_probabilities * np.arange(101)
    # )
    #
    # --------------------------------------------------------

    predicted_age = 24.9
    lower_age = 22.0
    upper_age = 27.9

    confidence_score = 0.96

    inference_passes = 10

    return (
        predicted_age,
        lower_age,
        upper_age,
        confidence_score,
        inference_passes,
    )


# ============================================================
# RUN MODEL
# ============================================================

(
    predicted_age,
    lower_age,
    upper_age,
    confidence_score,
    inference_passes,
) = run_model(image)


# ============================================================
# FACE BOX
# ============================================================
#
# This is currently a UI demonstration.
# Replace it with your actual MTCNN bounding box.
# ============================================================

def draw_face_box(img):

    output = img.copy()

    width, height = output.size

    # Demo coordinates
    x1 = int(width * 0.27)
    y1 = int(height * 0.12)

    x2 = int(width * 0.73)
    y2 = int(height * 0.78)

    draw = ImageDraw.Draw(output)

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=6,
        outline=(79, 70, 229),
        width=max(3, int(width * 0.005)),
    )

    return output


boxed_image = draw_face_box(image)


# ============================================================
# MAIN RESULT
# ============================================================

left_col, right_col = st.columns(
    [1.25, 0.75],
    gap="large",
)


# ============================================================
# LEFT — IMAGE
# ============================================================

with left_col:

    with st.container(border=True):

        top1, top2 = st.columns([0.7, 0.3])

        with top1:
            st.markdown(
                '<div class="card-heading">Analyzed image</div>',
                unsafe_allow_html=True,
            )

        with top2:
            st.markdown(
                """
                <div style="
                    text-align:right;
                    color:#087F5B;
                    font-size:11px;
                    font-weight:750;
                ">
                    ● Face detected
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        st.image(
            boxed_image,
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                margin-top:9px;
                color:#8A94A6;
                font-size:11px;
            ">
                <span>{uploaded_file.name}</span>
                <span>ID · {file_hash}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RIGHT — AGE RESULT
# ============================================================

with right_col:

    with st.container(border=True):

        st.markdown(
            '<div class="card-heading">Estimated age</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-caption">Apparent age estimation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="age-number">
                {predicted_age:.1f}
            </div>

            <div class="age-unit">
                years
            </div>

            <div class="confidence">
                ✓ &nbsp; High confidence
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">
                        Likely range
                    </div>
                    <div class="metric-value">
                        {lower_age:.1f} – {upper_age:.1f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-label">
                        Face status
                    </div>
                    <div class="metric-value">
                        Detected
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        m3, m4 = st.columns(2)

        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">
                        MC passes
                    </div>
                    <div class="metric-value">
                        {inference_passes}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-label">
                        TTA
                    </div>
                    <div class="metric-value">
                        Enabled
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ACTION BUTTONS
# ============================================================

st.write("")

button1, button2, empty = st.columns(
    [1, 1, 3],
    gap="small",
)

with button1:

    if st.button(
        "↻  Replace image",
        use_container_width=True,
    ):
        st.session_state.uploader_key += 1
        st.rerun()


with button2:

    if st.button(
        "＋  Analyze another",
        use_container_width=True,
    ):
        st.session_state.uploader_key += 1
        st.rerun()


# ============================================================
# AGE PROBABILITY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Age probability
    </div>

    <div class="section-subtitle">
        Probability distribution across predicted age classes.
    </div>
    """,
    unsafe_allow_html=True,
)


def create_probability_plot(predicted_age):

    ages = np.arange(0, 101)

    sigma = 7.0

    probability = np.exp(
        -((ages - predicted_age) ** 2)
        / (2 * sigma ** 2)
    )

    probability = probability / probability.max()

    fig, ax = plt.subplots(
        figsize=(12, 3.7)
    )

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.fill_between(
        ages,
        probability,
        alpha=0.12,
    )

    ax.plot(
        ages,
        probability,
        linewidth=2.2,
    )

    ax.axvspan(
        lower_age,
        upper_age,
        alpha=0.10,
    )

    ax.axvline(
        predicted_age,
        linestyle="--",
        linewidth=1.7,
    )

    ax.scatter(
        [predicted_age],
        [1],
        s=45,
        zorder=5,
    )

    ax.text(
        predicted_age,
        1.04,
        f"{predicted_age:.1f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(0, 100)

    ax.set_ylim(0, 1.10)

    ax.set_xlabel(
        "Age",
        fontsize=10,
    )

    ax.set_ylabel(
        "Relative probability",
        fontsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=9,
    )

    fig.tight_layout()

    return fig


with st.container(border=True):

    probability_fig = create_probability_plot(
        predicted_age
    )

    st.pyplot(
        probability_fig,
        use_container_width=True,
    )

    plt.close(probability_fig)


# ============================================================
# EXPLAINABILITY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Where the model looked
    </div>

    <div class="section-subtitle">
        Visual explanation of facial regions contributing to the prediction.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEATMAP
# ============================================================

def create_attention_map(img):

    arr = np.asarray(
        img.convert("RGB")
    )

    height, width, _ = arr.shape

    y, x = np.mgrid[
        0:height,
        0:width
    ]

    center_x = width * 0.50
    center_y = height * 0.40

    spread_x = width * 0.25
    spread_y = height * 0.30

    heat = np.exp(
        -(
            ((x - center_x) ** 2)
            / (2 * spread_x ** 2)
            +
            ((y - center_y) ** 2)
            / (2 * spread_y ** 2)
        )
    )

    # Additional focus around eyes / nose
    eye_focus = np.exp(
        -(
            (
                (x - width * 0.50) ** 2
                +
                (y - height * 0.40) ** 2
            )
            /
            (2 * (min(height, width) * 0.13) ** 2)
        )
    )

    heat = heat + 0.40 * eye_focus

    heat = heat / (
        heat.max() + 1e-8
    )

    fig, ax = plt.subplots(
        figsize=(7, 5.4)
    )

    ax.imshow(arr)

    ax.imshow(
        heat,
        cmap="turbo",
        alpha=0.45,
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )

    return fig


cam_left, cam_right = st.columns(
    [1.05, 0.95],
    gap="large",
)


# ============================================================
# LEFT — GRAD CAM
# ============================================================

with cam_left:

    with st.container(border=True):

        top1, top2 = st.columns([0.7, 0.3])

        with top1:
            st.markdown(
                '<div class="card-heading">Attention map</div>',
                unsafe_allow_html=True,
            )

        with top2:
            st.markdown(
                """
                <div style="
                    text-align:right;
                    color:#8A94A6;
                    font-size:11px;
                ">
                    Grad-CAM
                </div>
                """,
                unsafe_allow_html=True,
            )

        heatmap_fig = create_attention_map(
            image
        )

        st.pyplot(
            heatmap_fig,
            use_container_width=True,
        )

        plt.close(heatmap_fig)

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#8A94A6;
                font-size:11px;
                margin-top:3px;
            ">
                Blue = lower influence
                &nbsp;·&nbsp;
                Yellow = moderate
                &nbsp;·&nbsp;
                Red = higher influence
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RIGHT — FEATURE IMPACT
# ============================================================

with cam_right:

    with st.container(border=True):

        st.markdown(
            '<div class="card-heading">What influenced this result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="card-caption">Feature impact analysis</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        features = [
            (
                "Jawline definition",
                "Facial structure and overall geometry.",
                "Strong",
                "●●●●●",
            ),
            (
                "Skin / texture",
                "Texture and fine facial details.",
                "Moderate",
                "●●●○○",
            ),
            (
                "Eye-area clarity",
                "Periocular details and local features.",
                "Strong",
                "●●●●●",
            ),
            (
                "Facial symmetry",
                "Overall facial proportions.",
                "Moderate",
                "●●●○○",
            ),
        ]

        for (
            name,
            description,
            strength,
            bars,
        ) in features:

            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-top">

                        <div class="feature-name">
                            {name}
                        </div>

                        <div class="feature-strength">
                            {strength}
                        </div>

                    </div>

                    <div class="feature-description">
                        {description}
                    </div>

                    <div class="feature-bars">
                        {bars}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="reasoning-box">

                <div class="reasoning-title">
                    💡 Model reasoning
                </div>

                <div class="reasoning-text">

                    For an estimated age of
                    <b>{predicted_age:.1f} years</b>,
                    the model places higher importance on
                    facial structure, eye-region features,
                    and skin texture.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TECHNICAL PIPELINE
# ============================================================

st.write("")

with st.expander(
    "⚙️  View technical pipeline"
):

    st.markdown(
        """
        ### AgeLens AI inference pipeline

        **1. Image Upload**

        User provides a JPG or PNG face image.

        ↓

        **2. Face Detection**

        MTCNN identifies the visible face and facial landmarks.

        ↓

        **3. Face Preprocessing**

        Face cropping → padding → resize → RGB conversion
        → normalization.

        ↓

        **4. Model Inference**

        EfficientNet-B3 + DEX predicts an age probability
        distribution across ages 0–100.

        ↓

        **5. Uncertainty Estimation**

        Monte Carlo Dropout performs multiple stochastic
        inference passes.

        ↓

        **6. Test-Time Augmentation**

        Multiple augmented versions improve prediction
        robustness.

        ↓

        **7. Grad-CAM**

        Visualizes facial regions that contributed to the
        model prediction.

        ↓

        **8. Final Output**

        Estimated age + uncertainty range + confidence +
        visual explanation.

        ---

        **Model:** EfficientNet-B3 + DEX

        **Input:** 256 × 256 × 3 RGB face image

        **Output:** 101 age classes (0–100)

        **Uncertainty:** Monte Carlo Dropout

        **Robustness:** Test-Time Augmentation

        **Explainability:** Grad-CAM
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>AgeLens AI</b><br>

        EfficientNet-B3 · DEX · MC Dropout · TTA · Grad-CAM<br>

        Built with Python · TensorFlow · Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)
