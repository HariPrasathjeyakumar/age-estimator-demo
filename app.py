import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgeLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN PAGE ---------- */

    .stApp {
        background: #f7f9fc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 30px;
        padding-left: 40px;
        padding-right: 40px;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-logo {
        font-size: 25px;
        font-weight: 700;
        color: #111827;
        margin-top: 15px;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .sidebar-item {
        padding: 14px 8px;
        color: #334155;
        font-size: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }

    .sidebar-item:hover {
        background: #f1f5f9;
    }

    .model-ready {
        margin-top: 35px;
        padding: 18px;
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        background: white;
        font-weight: 600;
        color: #008f7a;
    }

    /* ---------- TITLE ---------- */

    .page-title {
        font-size: 38px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 3px;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 17px;
        margin-bottom: 25px;
    }

    /* ---------- PRIVACY ---------- */

    .privacy-box {
        border: 1px solid #cbdcf8;
        background: #f0f6ff;
        border-radius: 10px;
        padding: 14px 20px;
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 18px;
    }

    /* ---------- UPLOAD AREA ---------- */

    .upload-container {
        border: 1.5px dashed #b9c5d6;
        border-radius: 14px;
        padding: 12px;
        background: white;
        margin-bottom: 25px;
    }

    .upload-info {
        background: #252631;
        color: white;
        border-radius: 10px;
        padding: 20px;
    }

    .upload-title {
        font-size: 16px;
        font-weight: 600;
    }

    .upload-description {
        color: #cbd5e1;
        font-size: 14px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e1e7ef;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0px 2px 8px rgba(15, 23, 42, 0.03);
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 15px;
    }

    /* ---------- AGE ---------- */

    .age-label {
        color: #111827;
        font-size: 17px;
        font-weight: 650;
    }

    .age-number {
        font-size: 68px;
        font-weight: 400;
        color: #111827;
        line-height: 1;
        margin: 10px 0 18px 0;
    }

    .confidence {
        display: inline-block;
        background: #ecfdf5;
        color: #008f7a;
        border: 1px solid #b7eadc;
        padding: 9px 17px;
        border-radius: 25px;
        font-weight: 600;
        margin-bottom: 28px;
    }

    .range-title {
        color: #64748b;
        font-size: 15px;
    }

    .range-value {
        font-size: 25px;
        color: #111827;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ---------- INFO BOX ---------- */

    .info-box {
        background: #ffffff;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }

    .info-title {
        color: #334155;
        font-size: 15px;
        font-weight: 600;
    }

    .info-value {
        color: #111827;
        font-size: 18px;
        margin-top: 6px;
    }

    /* ---------- INFLUENCE ---------- */

    .influence-card {
        border: 1px solid #e1e7ef;
        border-radius: 12px;
        padding: 17px;
        margin-bottom: 12px;
        background: white;
    }

    .influence-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 16px;
    }

    .influence-description {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }

    .strength {
        color: #008f7a;
        font-weight: 600;
        text-align: right;
    }

    /* ---------- PIPELINE ---------- */

    .pipeline {
        background: #ffffff;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        padding: 20px;
        margin-top: 25px;
    }

    .pipeline-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-weight: 600;
        color: #334155;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 45px;
        margin-bottom: 20px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
        <div class="sidebar-logo">🧠 AgeLens AI</div>
        <div class="sidebar-subtitle">
            Intelligent age estimation
        </div>
    """, unsafe_allow_html=True)

    st.button(
        "🔄  New analysis",
        use_container_width=True
    )

    st.markdown("""
        <div class="sidebar-item">◷ &nbsp; History</div>
        <div class="sidebar-item">ⓘ &nbsp; About model</div>

        <div class="model-ready">
            🟢 &nbsp; Model ready
        </div>

        <div class="sidebar-item" style="margin-top:20px;">
            ⋯ &nbsp; Diagnostics
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-title">AgeLens AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Age estimation with uncertainty and visual explainability'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PRIVACY MESSAGE
# ============================================================

st.markdown("""
<div class="privacy-box">
    🛡️ &nbsp; Your image is processed for this analysis only.
</div>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("""
<div class="upload-info">
    <div class="upload-title">
        📤 &nbsp; Upload a face image
    </div>

    <div class="upload-description">
        Choose a clear JPG or PNG image containing a visible face.
        The AI will estimate apparent age and provide an explanation.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def estimate_age(image):
    """
    ---------------------------------------------------------
    REPLACE THIS FUNCTION WITH YOUR REAL MODEL.
    ---------------------------------------------------------

    Currently it returns a demo value.

    Example later:

        age = model.predict(face_image)

        return float(age)
    """

    return 24.9


def create_face_box(image):
    """
    Creates a demo face bounding box.

    Replace this with your MTCNN face detection output.
    """

    img = image.copy()

    width, height = img.size

    draw = ImageDraw.Draw(img)

    # Demo bounding box
    left = int(width * 0.25)
    top = int(height * 0.15)
    right = int(width * 0.75)
    bottom = int(height * 0.85)

    draw.rectangle(
        [left, top, right, bottom],
        outline="#4f46e5",
        width=4
    )

    return img


def create_heatmap(image):
    """
    Creates a visual demo attention map.

    Replace this with actual Grad-CAM output
    from your CNN/EfficientNet model.
    """

    img = image.convert("RGB")

    arr = np.array(img)

    # Create approximate attention region
    h, w, _ = arr.shape

    yy, xx = np.mgrid[0:h, 0:w]

    center_x = w * 0.50
    center_y = h * 0.42

    sigma_x = w * 0.22
    sigma_y = h * 0.25

    heat = np.exp(
        -(
            ((xx - center_x) ** 2 / (2 * sigma_x ** 2))
            +
            ((yy - center_y) ** 2 / (2 * sigma_y ** 2))
        )
    )

    heat = (heat * 255).astype(np.uint8)

    heat_img = Image.fromarray(heat).filter(
        ImageFilter.GaussianBlur(radius=20)
    )

    heat_array = np.array(heat_img) / 255.0

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(arr)
    ax.imshow(
        heat_array,
        cmap="jet",
        alpha=0.45
    )

    ax.axis("off")

    return fig


# ============================================================
# RESULT SECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    age = estimate_age(image)

    lower_age = age - 2.9
    upper_age = age + 3.0

    # --------------------------------------------------------
    # IMAGE + AGE
    # --------------------------------------------------------

    col1, col2 = st.columns([1.05, 1])

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                👤 &nbsp; Analyzed image
            </div>

        """, unsafe_allow_html=True)

        boxed_image = create_face_box(image)

        st.image(
            boxed_image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Buttons
        b1, b2 = st.columns(2)

        with b1:
            st.button(
                "▣  Replace image",
                use_container_width=True
            )

        with b2:
            st.button(
                "⟳  Analyze another",
                use_container_width=True
            )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        st.markdown("""
        <div class="card">

            <div class="age-label">
                Estimated age
            </div>

        """, unsafe_allow_html=True)

        st.markdown(
            f'<div class="age-number">{age:.1f}</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="confidence">
            ✓ &nbsp; High confidence
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)

        with r1:

            st.markdown("""
            <div class="range-title">
                📊 Likely range
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="range-value">
                    {lower_age:.1f} – {upper_age:.1f}
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.markdown("""
            <div class="range-title">
                🎯 Face status
            </div>

            <div class="range-value">
                Face detected
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


    # ========================================================
    # AGE PROBABILITY
    # ========================================================

    st.markdown("""
    <div class="card">

        <div class="section-title">
            Age probability
        </div>

        <div style="color:#64748b;">
            Probability distribution across predicted age classes.
        </div>

    </div>
    """, unsafe_allow_html=True)


    # Generate probability distribution
    ages = np.arange(0, 101)

    probabilities = np.exp(
        -((ages - age) ** 2) /
        (2 * 9 ** 2)
    )

    probabilities = probabilities / probabilities.sum()

    fig, ax = plt.subplots(figsize=(14, 4.5))

    ax.plot(
        ages,
        probabilities,
        linewidth=2
    )

    ax.fill_between(
        ages,
        probabilities,
        alpha=0.15
    )

    ax.axvline(
        age,
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Probability")

    ax.set_xlim(0, 100)

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown("""
    <div class="card">

        <div class="section-title">
            🔍 &nbsp; Where the model looked
        </div>

        <div style="color:#64748b;">
            Visual explanation of the facial regions contributing
            to the prediction.
        </div>

    </div>
    """, unsafe_allow_html=True)


    heat_col, influence_col = st.columns([1, 1])


    # ========================================================
    # HEATMAP
    # ========================================================

    with heat_col:

        st.markdown("""
        <h3>🔥 Attention map</h3>
        """, unsafe_allow_html=True)

        heatmap_fig = create_heatmap(image)

        st.pyplot(
            heatmap_fig,
            use_container_width=True
        )

        plt.close(heatmap_fig)

        st.markdown("""
        <div style="text-align:center;color:#64748b;">
            Blue = lower influence &nbsp;&nbsp;
            Yellow = moderate &nbsp;&nbsp;
            Red = higher influence
        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # INFLUENCE CARDS
    # ========================================================

    with influence_col:

        st.markdown("""
        <h3>🧬 &nbsp; What influenced this result</h3>
        """, unsafe_allow_html=True)

        # Card 1
        st.markdown("""
        <div class="influence-card">

            <div class="influence-title">
                ⌒ &nbsp; Facial structure
            </div>

            <div class="influence-description">
                Jawline definition, skin tautness,
                and periocular facial structure.
            </div>

            <div class="strength">
                Strong &nbsp; ●●●●●
            </div>

        </div>
        """, unsafe_allow_html=True)


        # Card 2
        st.markdown("""
        <div class="influence-card">

            <div class="influence-title">
                ◉ &nbsp; Skin / texture
            </div>

            <div class="influence-description">
                Skin texture and subtle facial
                characteristics contributed to the estimate.
            </div>

            <div class="strength">
                Moderate &nbsp; ●●●○○
            </div>

        </div>
        """, unsafe_allow_html=True)


        # Card 3
        st.markdown("""
        <div class="influence-card">

            <div class="influence-title">
                👁 &nbsp; Eye-area clarity
            </div>

            <div class="influence-description">
                Facial details around the eye region
                contribute to the learned representation.
            </div>

            <div class="strength">
                Strong &nbsp; ●●●●●
            </div>

        </div>
        """, unsafe_allow_html=True)


        # Card 4
        st.markdown("""
        <div class="influence-card">

            <div class="influence-title">
                ◡ &nbsp; Facial symmetry
            </div>

            <div class="influence-description">
                Overall facial proportions contribute
                to the estimated apparent age.
            </div>

            <div class="strength">
                Moderate &nbsp; ●●●○○
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # TECHNICAL PIPELINE
    # ========================================================

    with st.expander("⚙️ View technical pipeline"):

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:
            st.markdown("""
            <div class="pipeline-step">
                📤<br>
                Image Upload
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown("""
            <div class="pipeline-step">
                👤<br>
                MTCNN
            </div>
            """, unsafe_allow_html=True)

        with p3:
            st.markdown("""
            <div class="pipeline-step">
                🧠<br>
                CNN Model
            </div>
            """, unsafe_allow_html=True)

        with p4:
            st.markdown("""
            <div class="pipeline-step">
                📊<br>
                Age Prediction
            </div>
            """, unsafe_allow_html=True)

        with p5:
            st.markdown("""
            <div class="pipeline-step">
                🔥<br>
                Grad-CAM
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    AgeLens AI<br>

    EfficientNet-B3 • MC Dropout • TTA • Grad-CAM<br>

    Built with Python • TensorFlow • Streamlit

</div>
""", unsafe_allow_html=True)
