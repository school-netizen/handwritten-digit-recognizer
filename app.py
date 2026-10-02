import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf


# ============================================================
# PAGE CONFIG
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

    /* Main application background */
    .stApp {
        background-color: #0b1020;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Prediction number */
    .prediction-number {
        font-size: 7rem;
        font-weight: 900;
        text-align: center;
        line-height: 1;
        margin: 1rem 0;
    }

    /* Small prediction label */
    .prediction-label {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.15rem;
    }

    /* Confidence */
    .confidence-text {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Horizontal divider */
    hr {
        border-color: #263244;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #263244;
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

    st.error("❌ Unable to load the trained model.")

    with st.expander("Show error details"):
        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔢 DigitVision AI")

    st.write(
        "A CNN-based handwritten digit recognition system "
        "trained on the MNIST dataset."
    )

    st.divider()

    st.subheader("🧠 Model")

    st.write("**Type:** Convolutional Neural Network")

    st.write(
        """
        **Architecture**

        • Conv2D — 32 filters  
        • MaxPooling  
        • Conv2D — 64 filters  
        • MaxPooling  
        • Conv2D — 128 filters  
        • Flatten  
        • Dense — 128 neurons  
        • Dropout — 50%  
        • Output — 10 classes
        """
    )

    st.divider()

    st.subheader("📚 MNIST Dataset")

    st.write(
        """
        • 60,000 training images  
        • 10,000 test images  
        • 10 digit classes  
        • Image size: 28 × 28 pixels
        """
    )

    st.divider()

    st.subheader("⚙️ Image Processing")

    st.write(
        """
        Your drawing is processed using:

        1. Grayscale conversion
        2. Background detection
        3. Digit cropping
        4. Padding
        5. Aspect-ratio preserving resize
        6. Centering
        7. Normalization
        """
    )

    st.divider()

    st.caption("Built with TensorFlow + Streamlit")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔢 DigitVision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Draw a handwritten digit and let a convolutional neural
    network recognize it.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL STATUS
# ============================================================

if model_loaded:

    st.success("✅ CNN model loaded successfully.")

else:

    st.error(
        "The model could not be loaded. "
        "Make sure model/mnist_cnn.keras exists in your repository."
    )

    st.stop()


# ============================================================
# DRAWING SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">✏️ Draw a Digit</div>',
    unsafe_allow_html=True,
)

st.info(
    "Draw one digit from 0 to 9. For best results, make the digit "
    "large and keep it roughly centered."
)


canvas_col, prediction_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# CANVAS
# ============================================================

with canvas_col:

    st.subheader("Drawing Canvas")

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
# IMAGE PREPROCESSING
# ============================================================

def preprocess_canvas_image(canvas_image):
    """
    Convert the canvas drawing into an MNIST-style
    28x28 normalized image.

    Processing:

    RGBA
      ↓
    Grayscale
      ↓
    Find digit
      ↓
    Crop
      ↓
    Padding
      ↓
    Resize
      ↓
    Center
      ↓
    Normalize
    """

    if canvas_image is None:
        return None

    # Convert canvas image to PIL
    image = Image.fromarray(
        canvas_image.astype(np.uint8)
    )

    # RGBA -> grayscale
    image = image.convert("L")

    # PIL -> NumPy
    image_array = np.array(
        image,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Detect foreground
    # --------------------------------------------------------

    threshold = 20

    coords = np.argwhere(
        image_array > threshold
    )

    # No drawing
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # --------------------------------------------------------
    # Crop digit
    # --------------------------------------------------------

    cropped = image_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    if cropped.size == 0:
        return None

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    height, width = cropped.shape

    padding = max(
        10,
        int(max(height, width) * 0.20)
    )

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

    max_dimension = max(
        padded_image.width,
        padded_image.height
    )

    scale = 20 / max_dimension

    new_width = max(
        1,
        int(padded_image.width * scale)
    )

    new_height = max(
        1,
        int(padded_image.height * scale)
    )

    resized = padded_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    resized_array = np.array(
        resized,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Center on 28x28 image
    # --------------------------------------------------------

    final_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2

    final_image[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized_array

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = (
        final_image.astype(np.float32) / 255.0
    )

    return normalized


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(processed_image):

    # CNN expects:
    # (batch, height, width, channels)

    input_image = processed_image.reshape(
        1,
        28,
        28,
        1
    )

    probabilities = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_digit]
    )

    return (
        predicted_digit,
        confidence,
        probabilities
    )


# ============================================================
# PROCESS CURRENT DRAWING
# ============================================================

processed_image = None
predicted_digit = None
confidence = None
probabilities = None


if canvas.image_data is not None:

    processed_image = preprocess_canvas_image(
        canvas.image_data
    )

    if processed_image is not None:

        (
            predicted_digit,
            confidence,
            probabilities
        ) = predict_digit(
            processed_image
        )


# ============================================================
# PREDICTION DISPLAY
# ============================================================

with prediction_col:

    st.subheader("Prediction")

    if predicted_digit is not None:

        st.markdown(
            '<div class="prediction-label">Predicted Digit</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="prediction-number">{predicted_digit}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="confidence-text">'
            f'Confidence: <b>{confidence * 100:.2f}%</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.progress(
            float(confidence),
            text=f"Confidence: {confidence * 100:.2f}%"
        )

        # Second highest prediction
        sorted_indices = np.argsort(probabilities)

        second_digit = int(
            sorted_indices[-2]
        )

        second_confidence = float(
            probabilities[second_digit]
        )

        st.write(
            f"Second choice: **{second_digit}** "
            f"({second_confidence * 100:.2f}%)"
        )

    else:

        st.info(
            "✏️ Draw a digit on the canvas to see "
            "the prediction."
        )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

if probabilities is not None:

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '📊 Prediction Probabilities'
        '</div>',
        unsafe_allow_html=True,
    )

    probability_df = pd.DataFrame(
        {
            "Digit": [
                str(i)
                for i in range(10)
            ],
            "Probability": (
                probabilities * 100
            ),
        }
    )

    st.bar_chart(
        probability_df.set_index("Digit")[
            "Probability"
        ],
        height=320,
    )


# ============================================================
# IMAGE PREVIEW
# ============================================================

if processed_image is not None:

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        '🔍 Image Processing'
        '</div>',
        unsafe_allow_html=True,
    )

    original_col, processed_col = st.columns(
        2,
        gap="large"
    )

    with original_col:

        st.subheader("Original Drawing")

        original_image = Image.fromarray(
            canvas.image_data.astype(np.uint8)
        )

        st.image(
            original_image,
            width=300,
        )

    with processed_col:

        st.subheader("Processed 28 × 28")

        st.image(
            processed_image,
            width=300,
            clamp=True,
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">'
    '🧠 How It Works'
    '</div>',
    unsafe_allow_html=True,
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown("### 1️⃣ Draw")

    st.write(
        "Draw a handwritten digit using the canvas."
    )

with step2:

    st.markdown("### 2️⃣ Process")

    st.write(
        "The digit is cropped, padded, resized and centered."
    )

with step3:

    st.markdown("### 3️⃣ CNN")

    st.write(
        "The CNN extracts visual features from the image."
    )

with step4:

    st.markdown("### 4️⃣ Predict")

    st.write(
        "The network produces probabilities for digits 0–9."
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander("🔬 Technical Details"):

    st.write(
        """
        **Input shape:** 28 × 28 × 1

        **Output:** 10-class softmax probability distribution

        **Optimizer:** Adam

        **Loss:** Sparse categorical crossentropy

        **Dataset:** MNIST

        **Preprocessing:** Grayscale → Crop → Padding →
        Aspect-ratio resize → Center → Normalize
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔢 DigitVision AI • Handwritten Digit Recognition • "
    "TensorFlow + Streamlit + MNIST"
)
