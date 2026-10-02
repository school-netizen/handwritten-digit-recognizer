
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="DigitVision AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(99,102,241,0.15),
            transparent 35%
        ),
        radial-gradient(
            circle at top right,
            rgba(168,85,247,0.12),
            transparent 35%
        ),
        #0b1020;
}


/* Hero */

.hero {
    text-align: center;
    padding: 35px 10px 25px;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #818cf8,
            #c084fc,
            #f0abfc
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
}


/* Cards */

.card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
}


/* Prediction */

.prediction-card {
    text-align: center;

    padding: 30px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(99,102,241,0.20),
            rgba(168,85,247,0.10)
        );

    border: 1px solid
        rgba(129,140,248,0.25);
}

.prediction-digit {
    font-size: 7rem;
    font-weight: 900;
    line-height: 1;
    margin: 10px;
}

.confidence {
    color: #a5b4fc;
    font-size: 1.2rem;
    font-weight: 600;
}


/* Section */

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 15px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #080d1a;
}


/* Buttons */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/mnist_cnn.keras"
    )


model = load_model()


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🔢 DigitVision AI
</div>

<div class="hero-subtitle">
Handwritten Digit Recognition using a Convolutional Neural Network
</div>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🧠 About")

    st.write("""
    DigitVision AI uses a CNN trained on the
    MNIST handwritten digit dataset.
    """)

    st.divider()

    st.markdown("### 📊 Dataset")

    st.write("""
    **MNIST**

    • 60,000 training images  
    • 10,000 test images  
    • 28 × 28 pixels  
    • 10 digit classes
    """)

    st.divider()

    st.markdown("### 🧠 CNN")

    st.write("""
    **Architecture**

    Conv2D → Pooling → Conv2D → Pooling
    → Conv2D → Dense → Dropout → Softmax
    """)

    st.divider()

    st.markdown("### ✏️ Instructions")

    st.write("""
    1. Draw one digit.
    2. Keep it centered.
    3. Draw it fairly large.
    4. The model analyzes the image automatically.
    """)


# ==========================================================
# MAIN COLUMNS
# ==========================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ==========================================================
# DRAWING CANVAS
# ==========================================================

with left:

    st.markdown(
        '<div class="section-title">✏️ Draw a Digit</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use your mouse or trackpad to draw a digit from 0 to 9."
    )

    canvas = st_canvas(

        fill_color="rgba(0, 0, 0, 0)",

        stroke_width=18,

        stroke_color="#FFFFFF",

        background_color="#000000",

        height=350,

        width=350,

        drawing_mode="freedraw",

        display_toolbar=True,

        key="digit_canvas"
    )


# ==========================================================
# PROCESS IMAGE
# ==========================================================

prediction = None
processed_image = None


if canvas.image_data is not None:

    image = Image.fromarray(
        canvas.image_data.astype("uint8")
    ).convert("L")

    # MNIST uses white digits on black background

    image = ImageOps.invert(image)

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    processed_image = (
        np.array(image)
        .astype("float32")
        / 255.0
    )

    model_input = processed_image.reshape(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]


# ==========================================================
# PREDICTION
# ==========================================================

with right:

    st.markdown(
        '<div class="section-title">🤖 AI Prediction</div>',
        unsafe_allow_html=True
    )

    if prediction is not None:

        digit = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[digit]
        )

        st.markdown(
            f"""
            <div class="prediction-card">

            <div>
            The CNN predicts
            </div>

            <div class="prediction-digit">
            {digit}
            </div>

            <div class="confidence">
            {confidence * 100:.2f}% confidence
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        m1, m2 = st.columns(2)

        with m1:

            st.metric(
                "Predicted Digit",
                digit
            )

        with m2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%"
            )

        st.markdown("### 📊 Probability Distribution")

        probabilities = pd.DataFrame({
            "Digit": [
                str(i)
                for i in range(10)
            ],

            "Probability": [
                float(value) * 100
                for value in prediction
            ]
        })

        st.bar_chart(
            probabilities.set_index("Digit")
        )

    else:

        st.markdown("""
        <div class="prediction-card">

        <div style="font-size:4rem;">
        ✨
        </div>

        <h3>
        Ready for your digit
        </h3>

        <p style="color:#94a3b8;">
        Draw a digit on the canvas
        to see the CNN prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# IMAGE PROCESSING
# ==========================================================

if processed_image is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">🖼️ Image Processing</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            canvas.image_data,
            caption="Original Drawing"
        )

    with col2:

        st.image(
            processed_image,
            caption="28 × 28 MNIST Input",
            clamp=True
        )

    with col3:

        st.markdown("""
        <div class="card">

        <b>Preprocessing</b>

        <br><br>

        ✓ Grayscale conversion

        <br><br>

        ✓ Color inversion

        <br><br>

        ✓ Resize to 28 × 28

        <br><br>

        ✓ Pixel normalization

        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# HOW IT WORKS
# ==========================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 How the CNN Works</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown("""
    <div class="card">

    ### 1️⃣ Input

    28 × 28 grayscale image.

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="card">

    ### 2️⃣ Features

    Convolution layers
    detect visual patterns.

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="card">

    ### 3️⃣ Classification

    Dense layers combine
    learned features.

    </div>
    """, unsafe_allow_html=True)


with c4:

    st.markdown("""
    <div class="card">

    ### 4️⃣ Output

    Softmax produces
    probabilities for 0–9.

    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

DigitVision AI · MNIST CNN

<br>

Built with TensorFlow + Streamlit

</div>
""", unsafe_allow_html=True)
