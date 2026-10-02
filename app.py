import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DigitVision AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.10), transparent 30%),
                #0b1020;
            color: #f8fafc;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    rgba(79, 70, 229, 0.22),
                    rgba(124, 58, 237, 0.15)
                );
            border: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.3rem;
            font-weight: 800;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.1rem;
            margin-bottom: 0;
        }

        .card {
            padding: 1.4rem;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }

        .prediction-card {
            text-align: center;
            padding: 2rem;
            border-radius: 24px;
            background:
                linear-gradient(
                    145deg,
                    rgba(79, 70, 229, 0.25),
                    rgba(124, 58, 237, 0.18)
                );
            border: 1px solid rgba(255,255,255,0.12);
        }

        .prediction-label {
            color: #94a3b8;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .prediction-digit {
            font-size: 7rem;
            line-height: 1;
            font-weight: 900;
            margin: 0.7rem 0;
        }

        .confidence {
            font-size: 1.25rem;
            color: #cbd5e1;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        .info-box {
            padding: 1rem;
            border-radius: 14px;
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 0.8rem;
        }

        .step-card {
            padding: 1.2rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255,255,255,0.08);
            height: 100%;
        }

        .step-number {
            font-size: 1.6rem;
            font-weight: 800;
        }

        .footer {
            text-align: center;
            color: #64748b;
            padding-top: 2rem;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.65);
            padding: 1rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.07);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mnist_cnn.keras")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    st.error("Unable to load the trained model.")
    st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔢 DigitVision AI")

    st.markdown(
        """
        Draw a handwritten digit and let the CNN
        predict what you wrote.
        """
    )

    st.divider()

    st.markdown("### 🧠 Model")

    st.markdown(
        """
        **Architecture**

        • Convolutional Layer — 32 filters  
        • Max Pooling  
        • Convolutional Layer — 64 filters  
        • Max Pooling  
        • Convolutional Layer — 128 filters  
        • Dense Layer — 128 neurons  
        • Dropout — 50%  
        • Output — 10 classes
        """
    )

    st.divider()

    st.markdown("### 📚 Dataset")

    st.markdown(
        """
        **MNIST**

        • 60,000 training images  
        • 10,000 test images  
        • 10 digit classes  
        • Image size: 28 × 28 pixels
        """
    )

    st.divider()

    st.markdown("### ⚙️ Preprocessing")

    st.markdown(
        """
        Your drawing is:

        1. Converted to grayscale
        2. Cropped around the digit
        3. Padded to preserve shape
        4. Resized to 28 × 28
        5. Normalized to 0–1
        6. Sent to the CNN
        """
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔢 DigitVision AI</h1>
        <p>
            Draw a handwritten digit and watch a convolutional
            neural network recognize it in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_canvas_image(canvas_image):
    """
    Convert the canvas RGBA image into an MNIST-like
    28x28 grayscale image.

    Steps:
    1. Convert RGBA -> grayscale
    2. Find non-empty pixels
    3. Crop around the digit
    4. Add padding
    5. Resize while preserving aspect ratio
    6. Place digit in the center of a 28x28 canvas
    7. Normalize to [0, 1]
    """

    if canvas_image is None:
        return None

    # Convert to PIL image
    image = Image.fromarray(canvas_image.astype(np.uint8))

    # Convert RGBA to grayscale
    image = image.convert("L")

    # Convert to numpy
    image_array = np.array(image, dtype=np.uint8)

    # --------------------------------------------------------
    # Find foreground
    # --------------------------------------------------------

    # Canvas is black and drawing is white.
    # Anything above a small threshold is considered foreground.
    threshold = 20

    coords = np.argwhere(image_array > threshold)

    # No drawing
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # --------------------------------------------------------
    # Crop around digit
    # --------------------------------------------------------

    cropped = image_array[y_min:y_max + 1, x_min:x_max + 1]

    if cropped.size == 0:
        return None

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    height, width = cropped.shape

    padding = max(10, int(max(height, width) * 0.20))

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding

    padded = np.zeros(
        (padded_height, padded_width),
        dtype=np.uint8
    )

    padded[
        padding:padding + height,
        padding:padding + width
    ] = cropped

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    padded_image = Image.fromarray(padded)

    max_dimension = max(padded_image.size)

    scale = 20 / max_dimension

    new_width = max(1, int(padded_image.width * scale))
    new_height = max(1, int(padded_image.height * scale))

    resized = padded_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    resized_array = np.array(resized, dtype=np.uint8)

    # --------------------------------------------------------
    # Center on 28x28 MNIST-style canvas
    # --------------------------------------------------------

    final_image = np.zeros((28, 28), dtype=np.uint8)

    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2

    final_image[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized_array

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = final_image.astype(np.float32) / 255.0

    return normalized


def predict_digit(processed_image):
    """
    Run the processed 28x28 image through the CNN.
    """

    input_image = processed_image.reshape(1, 28, 28, 1)

    probabilities = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted_digit = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_digit])

    return predicted_digit, confidence, probabilities


# ============================================================
# DRAWING AREA
# ============================================================

st.markdown(
    '<div class="section-title">✏️ Draw Your Digit</div>',
    unsafe_allow_html=True
)

st.info(
    "Draw a single digit from 0 to 9 using your mouse or trackpad. "
    "For best results, draw it large and near the center."
)


left, right = st.columns([1.05, 0.95], gap="large")


with left:

    st.markdown(
        """
        <div class="card">
        <b>Canvas</b><br>
        <span style="color:#94a3b8;">
        White strokes on a dark background work best.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # display_toolbar has intentionally been removed because
    # it is not supported by the installed version of
    # streamlit-drawable-canvas.

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=14,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=350,
        width=350,
        drawing_mode="freedraw",
        return_image_data=True,
        key="digit_canvas",
    )


# ============================================================
# PROCESS DRAWING
# ============================================================

processed_image = None
predicted_digit = None
confidence = None
probabilities = None


if canvas.image_data is not None and model_loaded:

    processed_image = preprocess_canvas_image(
        canvas.image_data
    )

    if processed_image is not None:

        predicted_digit, confidence, probabilities = predict_digit(
            processed_image
        )


# ============================================================
# PREDICTION RESULT
# ============================================================

with right:

    if predicted_digit is not None:

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Digit
                </div>

                <div class="prediction-digit">
                    {predicted_digit}
                </div>

                <div class="confidence">
                    Confidence: <b>{confidence * 100:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence),
            text=f"Confidence: {confidence * 100:.2f}%"
        )

    else:

        st.markdown(
            """
            <div class="prediction-card">
                <div class="prediction-label">
                    Prediction
                </div>

                <div class="prediction-digit">
                    ?
                </div>

                <div class="confidence">
                    Draw a digit to begin
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# RESULTS
# ============================================================

if predicted_digit is not None:

    st.markdown(
        '<div class="section-title">📊 Prediction Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Digit",
            predicted_digit
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col3:
        second_best = np.argsort(probabilities)[-2]
        second_confidence = probabilities[second_best]

        st.metric(
            "Second Choice",
            f"{second_best} ({second_confidence * 100:.1f}%)"
        )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

if probabilities is not None:

    st.markdown(
        '<div class="section-title">📈 Probability Distribution</div>',
        unsafe_allow_html=True
    )

    probability_df = pd.DataFrame(
        {
            "Digit": [str(i) for i in range(10)],
            "Probability": probabilities
        }
    )

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )

    st.bar_chart(
        probability_df.set_index("Digit")["Probability"],
        height=300
    )


# ============================================================
# IMAGE PROCESSING PREVIEW
# ============================================================

if processed_image is not None:

    st.markdown(
        '<div class="section-title">🔍 What the CNN Sees</div>',
        unsafe_allow_html=True
    )

    preview_col1, preview_col2 = st.columns(2)

    with preview_col1:

        st.markdown(
            """
            <div class="card">
            <b>Original Drawing</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        original_image = Image.fromarray(
            canvas.image_data.astype(np.uint8)
        )

        st.image(
            original_image,
            width=280
        )

    with preview_col2:

        st.markdown(
            """
            <div class="card">
            <b>Processed 28 × 28 Image</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            processed_image,
            width=280,
            clamp=True
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 How DigitVision AI Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">01</div>
            <h4>Draw</h4>
            <p>
            Draw a handwritten digit on the canvas.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step2:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">02</div>
            <h4>Process</h4>
            <p>
            The drawing is cropped, centered,
            resized and normalized.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step3:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">03</div>
            <h4>CNN</h4>
            <p>
            The convolutional neural network
            analyzes visual features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with step4:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">04</div>
            <h4>Prediction</h4>
            <p>
            The model outputs probabilities
            for all ten digits.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("Input", "28 × 28")

with info2:
    st.metric("Classes", "10")

with info3:
    st.metric("Dataset", "MNIST")

with info4:
    st.metric("Model", "CNN")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using TensorFlow, Streamlit and MNIST
        <br>
        DigitVision AI — Handwritten Digit Recognition
    </div>
    """,
    unsafe_allow_html=True
)
