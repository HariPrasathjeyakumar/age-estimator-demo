import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import textwrap

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AgeLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML RENDER HELPER  <-- THE FIX
# ============================================================
# st.markdown() runs its input through a Markdown parser first.
# Markdown treats ANY line indented 4+ spaces as an indented code
# block and renders it as literal text - unsafe_allow_html=True
# does not prevent this, because the content isn't being blocked,
# it's being converted into a valid <pre><code> block containing
# your literal tag text (this app is especially exposed to this
# because of the multi-line inline `style="...long css..."`
# blocks, where every CSS property line is indented).
#
# render_html() strips leading whitespace from EVERY line
# individually (not just the common prefix, since outer tags are
# often flush-left while inner tags/lines are indented for
# readability) before handing the string to st.markdown(), no
# matter how deeply the call is nested in your Python code.
# Used for both raw HTML and multi-line markdown text, since
# plain **bold** markdown text is just as vulnerable to the same
# indented-code-block problem.

def render_html(html: str):
    lines = [line.lstrip() for line in html.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background: #F7F8FC;
        color: #172033;
    }

    .main {
        padding: 0 !important;
    }

    .block-container {
        max-width: 1400px;
        padding: 2rem 3rem 4rem 3rem;
    }

    h1, h2, h3, h4, p {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E6E9F0;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1.2rem;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .logo-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(
            135deg,
            #5B4BFF,
            #7C5CFC
        );
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(91,75,255,0.20);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 750;
        color: #172033;
    }

    .sidebar-subtitle {
        color: #7A8499;
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .sidebar-section {
        font-size: 10px;
        font-weight: 800;
        color: #9AA3B5;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 22px 0 10px 2px;
    }

    .status-card {
        margin-top: 35px;
        padding: 16px;
        border: 1px solid #E5E8F0;
        border-radius: 14px;
        background: #FBFCFE;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 700;
        color: #263248;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        background: #18B981;
        border-radius: 50%;
    }

    .status-text {
        color: #8993A6;
        font-size: 11px;
        margin-top: 8px;
    }

    /* -------------------------------------------------------
       HEADER
    ------------------------------------------------------- */

    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 25px;
    }

    .page-title {
        font-size: 36px;
        line-height: 1.15;
        font-weight: 800;
        color: #121A2A;
        margin: 0;
        letter-spacing: -1px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #7A8499;
        margin-top: 8px;
    }

    .model-ready {
        background: #FFFFFF;
        border: 1px solid #E5E8F0;
        border-radius: 12px;
        padding: 13px 18px;
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 13px;
        font-weight: 700;
        color: #263248;
        box-shadow: 0 3px 12px rgba(25,35,55,0.04);
    }

    .ready-dot {
        width: 8px;
        height: 8px;
        background: #18B981;
        border-radius: 50%;
    }

    /* -------------------------------------------------------
       PRIVACY BANNER
    ------------------------------------------------------- */

    .privacy-banner {
        background: #EFF5FF;
        border: 1px solid #CFE0FF;
        border-radius: 12px;
        padding: 14px 18px;
        color: #315C9B;
        font-size: 13px;
        margin: 20px 0 30px 0;
    }

    /* -------------------------------------------------------
       SECTION
    ------------------------------------------------------- */

    .section-title {
        font-size: 19px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 4px;
    }

    .section-description {
        color: #7B8599;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* -------------------------------------------------------
       UPLOAD
    ------------------------------------------------------- */

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1.5px dashed #BFC8DD;
        border-radius: 16px;
        padding: 12px;
        transition: 0.2s ease;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #695BFF;
        background: #FBFAFF;
    }

    div[data-testid="stFileUploader"] section {
        border: none !important;
    }

    /* -------------------------------------------------------
       CARDS
    ------------------------------------------------------- */

    .card {
        background: #FFFFFF;
        border: 1px solid #E5E8F0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 20px rgba(24, 34, 55, 0.035);
    }

    .card-title {
        font-size: 17px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #8A93A6;
        font-size: 12px;
    }

    /* -------------------------------------------------------
       IMAGE
    ------------------------------------------------------- */

    .image-container {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #E4E7EE;
        background: #F4F5F8;
    }

    /* -------------------------------------------------------
       AGE RESULT
    ------------------------------------------------------- */

    .age-label {
        color: #7B8599;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    .age-value {
        font-size: 62px;
        line-height: 1;
        font-weight: 800;
        color: #121A2A;
        letter-spacing: -2px;
    }

    .age-unit {
        color: #7B8599;
        font-size: 14px;
        margin-top: 8px;
    }

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 18px;
        padding: 8px 12px;
        border-radius: 999px;
        background: #ECFDF6;
        border: 1px solid #C7F2DF;
        color: #12805B;
        font-size: 12px;
        font-weight: 700;
    }

    /* -------------------------------------------------------
       METRIC CARDS
    ------------------------------------------------------- */

    .metric-box {
        background: #F9FAFC;
        border: 1px solid #E7EAF0;
        border-radius: 12px;
        padding: 15px;
        margin-top: 18px;
    }

    .metric-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #8D96A8;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 16px;
        font-weight: 750;
        color: #253047;
        margin-top: 6px;
    }

    /* -------------------------------------------------------
       FEATURE CARDS
    ------------------------------------------------------- */

    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E5E8F0;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .feature-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .feature-name {
        font-size: 14px;
        font-weight: 750;
        color: #263248;
    }

    .feature-strength {
        font-size: 11px;
        font-weight: 750;
        color: #5B4BFF;
    }

    .feature-description {
        color: #7D879A;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.5;
    }

    .feature-bar {
        height: 5px;
        background: #EDEFF5;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 12px;
    }

    .feature-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #5B4BFF,
            #8B7DFF
        );
        border-radius: 20px;
    }

    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #DDE1EA;
        background: #FFFFFF;
        color: #263248;
        font-weight: 650;
        height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #6658FF;
        color: #5848F5;
        background: #F9F8FF;
    }

    /* -------------------------------------------------------
       EXPANDER
    ------------------------------------------------------- */

    div[data-testid="stExpander"] {
        border: 1px solid #E3E6EE !important;
        border-radius: 14px !important;
        background: #FFFFFF !important;
    }

    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #9AA3B4;
        font-size: 11px;
        padding: 35px 0 10px 0;
    }

    /* -------------------------------------------------------
       HIDE STREAMLIT DEFAULT ELEMENTS
    ------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
        <div class="sidebar-logo">
            <div class="logo-icon">✦</div>
            <div class="sidebar-title">AgeLens AI</div>
        </div>
        <div class="sidebar-subtitle">
            Intelligent age estimation
        </div>
    """)

    render_html('<div class="sidebar-section">Workspace</div>')

    if st.button("＋  New analysis", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if st.button("◷  History", use_container_width=True):
        st.info("Analysis history will appear here.")

    if st.button("ⓘ  About model", use_container_width=True):
        st.info(
            "AgeLens AI uses a deep learning model for "
            "facial age estimation with uncertainty "
            "and visual explainability."
        )

    render_html('<div class="sidebar-section">System</div>')

    if st.button("⌁  Diagnostics", use_container_width=True):
        st.info("Model status: Ready")

    render_html("""
        <div class="status-card">
            <div class="status-row">
                <div class="status-dot"></div>
                Model ready
            </div>
            <div class="status-text">
                EfficientNet-B3 · Production mode
            </div>
        </div>
    """)


# ============================================================
# HEADER
# ============================================================

render_html("""
<div class="page-header">

    <div>
        <div class="page-title">AgeLens AI</div>
        <div class="page-subtitle">
            Age estimation with uncertainty and visual explainability
        </div>
    </div>

    <div class="model-ready">
        <div class="ready-dot"></div>
        Model ready
    </div>

</div>
""")


# ============================================================
# PRIVACY MESSAGE
# ============================================================

render_html("""
<div class="privacy-banner">
    🔒 &nbsp;
    <b>Your image is processed for this analysis only.</b>
    &nbsp; Images are not retained by the application.
</div>
""")


# ============================================================
# UPLOAD SECTION
# ============================================================

render_html('<div class="section-title">Analyze a face image</div>')

render_html("""
<div class="section-description">
    Upload a clear JPG or PNG image containing a visible face.
    The AI will estimate apparent age and provide visual explanations.
</div>
""")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# MODEL / INFERENCE SECTION
# ============================================================

def run_age_estimation(image):
    """
    ============================================================
    CONNECT YOUR EXISTING MODEL HERE
    ============================================================

    Replace this function body with your current:
        - MTCNN face detection
        - preprocessing
        - EfficientNet-B3 inference
        - MC Dropout
        - TTA
        - Grad-CAM

    It must return:

        estimated_age
        confidence
        lower_age
        upper_age
        probability_distribution
        gradcam_image
        face_detected

    """

    # --------------------------------------------------------
    # TEMPORARY DEMO OUTPUT
    # --------------------------------------------------------
    #
    # DELETE THIS SECTION when connecting your trained model.
    #

    estimated_age = 24.9
    confidence = 0.91
    lower_age = 22.0
    upper_age = 27.9

    ages = np.arange(101)

    probabilities = np.exp(
        -0.5 * ((ages - estimated_age) / 10) ** 2
    )

    probabilities = probabilities / probabilities.sum()

    gradcam_image = image.copy()

    face_detected = True

    return (
        estimated_age,
        confidence,
        lower_age,
        upper_age,
        probabilities,
        gradcam_image,
        face_detected
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing facial features..."):

        (
            estimated_age,
            confidence,
            lower_age,
            upper_age,
            probabilities,
            gradcam_image,
            face_detected
        ) = run_age_estimation(image)

    render_html("<br>")

    # ========================================================
    # RESULT SECTION
    # ========================================================

    left, right = st.columns(
        [1.65, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        render_html("""
        <div class="card-title">
            Analyzed image
        </div>
        """)

        render_html(
            """
            <div style="
                display:flex;
                justify-content:flex-end;
                margin-bottom:8px;
            ">
                <span style="
                    color:#14805A;
                    font-size:12px;
                    font-weight:700;
                ">
                    ● Face detected
                </span>
            </div>
            """
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # AGE RESULT
    # --------------------------------------------------------

    with right:

        render_html('<div class="card">')

        render_html('<div class="age-label">Estimated age</div>')

        render_html(f'<div class="age-value">{estimated_age:.1f}</div>')

        render_html('<div class="age-unit">years</div>')

        confidence_percentage = confidence * 100

        render_html(
            f"""
            <div class="confidence-badge">
                ✓ &nbsp; High confidence · {confidence_percentage:.0f}%
            </div>
            """
        )

        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">
                Likely range
            </div>
            <div class="metric-value">
                {lower_age:.1f} – {upper_age:.1f} years
            </div>
        </div>
        """)

        render_html(f"""
        <div class="metric-box">
            <div class="metric-label">
                Face status
            </div>
            <div class="metric-value">
                {"Face detected" if face_detected else "No face detected"}
            </div>
        </div>
        """)

        render_html("</div>")


    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    render_html("<br>")

    col1, col2, col3 = st.columns(
        [1, 1, 2]
    )

    with col1:
        if st.button(
            "↻  Analyze another",
            use_container_width=True
        ):
            st.rerun()

    with col2:
        if st.button(
            "▣  Replace image",
            use_container_width=True
        ):
            st.rerun()


    # ========================================================
    # AGE PROBABILITY
    # ========================================================

    render_html("<br>")

    render_html('<div class="section-title">Age probability</div>')

    render_html("""
    <div class="section-description">
        Probability distribution across predicted age classes.
    </div>
    """)

    fig, ax = plt.subplots(
        figsize=(12, 4.2)
    )

    ages = np.arange(len(probabilities))

    ax.plot(
        ages,
        probabilities,
        linewidth=2.2
    )

    ax.fill_between(
        ages,
        probabilities,
        alpha=0.12
    )

    ax.axvline(
        estimated_age,
        linestyle="--",
        linewidth=1.8
    )

    ax.axvspan(
        lower_age,
        upper_age,
        alpha=0.10
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel(
        "Age",
        fontsize=10
    )

    ax.set_ylabel(
        "Probability",
        fontsize=10
    )

    ax.grid(
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    render_html("<br>")

    render_html('<div class="section-title">Visual explainability</div>')

    render_html("""
    <div class="section-description">
        Understand which facial regions contributed to the prediction.
    </div>
    """)

    explain_left, explain_right = st.columns(
        [1.15, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # GRAD CAM
    # --------------------------------------------------------

    with explain_left:

        render_html("""
        <div class="card-title">
            Attention map
        </div>
        <div class="card-subtitle">
            Grad-CAM visual explanation
        </div>
        """)

        st.image(
            gradcam_image,
            use_container_width=True
        )

        render_html("""
        <div style="
            text-align:center;
            color:#7B8599;
            font-size:11px;
            margin-top:8px;
        ">
            Blue = lower influence &nbsp;&nbsp;
            Yellow = moderate &nbsp;&nbsp;
            Red = higher influence
        </div>
        """)


    # --------------------------------------------------------
    # FEATURE IMPACT
    # --------------------------------------------------------

    with explain_right:

        render_html("""
        <div class="card-title">
            What influenced this result
        </div>

        <div class="card-subtitle">
            Feature impact analysis
        </div>
        """)

        features = [
            (
                "Facial structure",
                "Strong",
                "Jawline definition and overall facial geometry.",
                88
            ),
            (
                "Skin / texture",
                "Moderate",
                "Texture patterns contributing to apparent age.",
                68
            ),
            (
                "Eye region",
                "Strong",
                "Facial details around the eye region.",
                82
            ),
            (
                "Facial symmetry",
                "Moderate",
                "Overall facial proportions influencing the estimate.",
                61
            )
        ]

        for name, strength, description, percentage in features:

            render_html(f"""
            <div class="feature-card">

                <div class="feature-header">

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

                <div class="feature-bar">
                    <div
                        class="feature-fill"
                        style="width:{percentage}%;">
                    </div>
                </div>

            </div>
            """)


    # ========================================================
    # MODEL REASONING
    # ========================================================

    render_html("<br>")

    with st.expander(
        "🔬  View model reasoning",
        expanded=False
    ):

        render_html(f"""
        **Model interpretation**

        For an estimated age of **{estimated_age:.1f} years**, 
        the model placed higher importance on facial geometry,
        skin texture and periocular facial features.

        The prediction is based on learned visual representations
        from the trained neural network and should be interpreted
        as an estimate rather than an exact biological age.
        """)


    # ========================================================
    # TECHNICAL PIPELINE
    # ========================================================

    with st.expander(
        "⚙️  View technical pipeline",
        expanded=False
    ):

        pipeline_cols = st.columns(5)

        pipeline = [
            ("01", "Input", "Face image"),
            ("02", "Detection", "Face localization"),
            ("03", "Preprocessing", "Resize & normalize"),
            ("04", "Inference", "EfficientNet-B3"),
            ("05", "Explainability", "MC Dropout + Grad-CAM")
        ]

        for col, (number, title, description) in zip(
            pipeline_cols,
            pipeline
        ):

            with col:

                render_html(f"""
                <div style="
                    background:#F9FAFC;
                    border:1px solid #E5E8F0;
                    border-radius:12px;
                    padding:15px;
                    min-height:115px;
                ">

                    <div style="
                        color:#6658FF;
                        font-size:11px;
                        font-weight:800;
                    ">
                        {number}
                    </div>

                    <div style="
                        color:#263248;
                        font-weight:750;
                        font-size:13px;
                        margin-top:7px;
                    ">
                        {title}
                    </div>

                    <div style="
                        color:#8A93A6;
                        font-size:11px;
                        margin-top:6px;
                        line-height:1.4;
                    ">
                        {description}
                    </div>

                </div>
                """)


# ============================================================
# EMPTY STATE
# ============================================================

else:

    render_html("<br>")

    empty_left, empty_center, empty_right = st.columns(
        [1, 2, 1]
    )

    with empty_center:

        render_html("""
        <div style="
            background:#FFFFFF;
            border:1px solid #E5E8F0;
            border-radius:18px;
            padding:55px 35px;
            text-align:center;
        ">

            <div style="
                width:65px;
                height:65px;
                border-radius:18px;
                background:#F0EEFF;
                color:#5B4BFF;
                display:flex;
                align-items:center;
                justify-content:center;
                margin:0 auto 20px auto;
                font-size:30px;
            ">
                ◎
            </div>

            <div style="
                font-size:20px;
                font-weight:750;
                color:#172033;
            ">
                Upload a face image
            </div>

            <div style="
                color:#7B8599;
                font-size:13px;
                line-height:1.6;
                margin-top:9px;
            ">
                Choose a clear JPG or PNG image containing
                a visible face to begin age estimation.
            </div>

        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    <b>AgeLens AI</b>
    &nbsp; · &nbsp;
    EfficientNet-B3
    &nbsp; · &nbsp;
    MC Dropout
    &nbsp; · &nbsp;
    TTA
    &nbsp; · &nbsp;
    Grad-CAM
    <br>
    Built with Python · TensorFlow · Streamlit
</div>
""")
