import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="EcoClassify AI - Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.subtitle {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 3rem;
}
.prediction-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #2E8B57;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="main-header">♻️ EcoClassify AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Waste Classification System</p>', unsafe_allow_html=True)

# Load TFLite model with caching
@st.cache_resource
def load_tflite_model(model_path='waste_classifier.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Define classes
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load the model interpreter
interpreter = load_tflite_model()

# Recycling tips dictionary
RECYCLING_TIPS = {
    'cardboard': {
        'icon': '📦',
        'tips': [
            "Remove all tape and staples before recycling",
            "Flatten boxes to save space",
            "Keep cardboard dry and clean",
            "Pizza boxes can be recycled if not too greasy"
        ],
        'color': '#8B4513'
    },
    'glass': {
        'icon': '🥛',
        'tips': [
            "Rinse containers before recycling",
            "Remove caps and lids",
            "Separate by color if required locally",
            "Don't include broken window glass"
        ],
        'color': '#20B2AA'
    },
    'metal': {
        'icon': '🔧',
        'tips': [
            "Clean cans and containers",
            "Remove labels if possible",
            "Aluminum cans are infinitely recyclable",
            "Steel cans are magnetic and recyclable"
        ],
        'color': '#708090'
    },
    'paper': {
        'icon': '📄',
        'tips': [
            "Keep paper clean and dry",
            "Remove plastic components",
            "Shredded paper goes in clear bags",
            "Avoid recycling waxed or plastic-coated paper"
        ],
        'color': '#F5F5DC'
    },
    'plastic': {
        'icon': '🥤',
        'tips': [
            "Check the recycling number (1-7)",
            "Rinse containers thoroughly",
            "Remove caps and lids (recycle separately)",
            "Avoid plastic bags in regular recycling"
        ],
        'color': '#87CEEB'
    },
    'trash': {
        'icon': '🗑️',
        'tips': [
            "Consider if item can be repurposed",
            "Check for special disposal requirements",
            "Some 'trash' items have special recycling programs",
            "Reduce waste by choosing reusable alternatives"
        ],
        'color': '#696969'
    }
}

# Main interface layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload or Take a Photo")

    camera_image = st.camera_input("Or take a photo with your camera")
    uploaded_file = st.file_uploader(
        "Choose an image of waste to classify",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the waste item for best results"
    )
    image = None

    if camera_image is not None:
        image = Image.open(camera_image)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)

    if image is not None:
        st.image(image, caption='Selected Image', width='stretch')

        if st.button("🔍 Classify Waste", type="primary"):
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                predictions = predict_tflite(interpreter, processed_image)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                st.session_state.prediction = predicted_class
                st.session_state.confidence = confidence
                st.session_state.all_predictions = predictions[0]

with col2:
    st.header("📊 Classification Results")

    if hasattr(st.session_state, 'prediction'):
        predicted_class = st.session_state.prediction
        confidence = st.session_state.confidence
        all_predictions = st.session_state.all_predictions

        tips = RECYCLING_TIPS[predicted_class]
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: {tips['color']}; margin: 0;">
                {tips['icon']} {predicted_class.title()}
            </h2>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                Confidence: <strong>{confidence*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("All Predictions")
        prediction_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Confidence': all_predictions * 100
        }).sort_values('Confidence', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(prediction_df['Class'], prediction_df['Confidence'])
        max_idx = prediction_df['Confidence'].idxmax()
        bars[max_idx].set_color('#2E8B57')
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Classification Confidence for All Classes')
        ax.set_xlim(0, 100)
        for i, (idx, row) in enumerate(prediction_df.iterrows()):
            ax.text(row['Confidence'] + 1, i, f'{row["Confidence"]:.1f}%', 
                   va='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader(f"♻️ Recycling Tips for {predicted_class.title()}")
        for tip in tips['tips']:
            st.markdown(f"• {tip}")

        st.subheader("🌍 Environmental Impact")
        if predicted_class in ['cardboard', 'glass', 'metal', 'paper', 'plastic']:
            st.success("✅ This item is recyclable! Proper recycling helps reduce landfill waste and conserves resources.")
        else:
            st.warning("⚠️ This item may not be recyclable through standard programs. Consider reuse or special disposal methods.")
    else:
        st.info("👆 Upload an image or take a photo above to get started with waste classification!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>EcoClassify AI - Powered by TensorFlow Lite & EfficientNetV2B1</p>
    <p>Help protect our environment through proper waste classification! 🌍</p>
</div>
""", unsafe_allow_html=True)

# Sidebar information
st.sidebar.header("About EcoClassify AI")
st.sidebar.info("""
This AI system can classify waste into different categories:
- 📦 Cardboard
- 🥛 Glass  
- 🔧 Metal
- 📄 Paper
- 🥤 Plastic
- 🗑️ Trash
Upload an image or take a photo to get started!
""")
st.sidebar.header("How it works")
st.sidebar.markdown("""
1. **Upload** an image of waste or **take a photo** using your camera
2. **AI Analysis** using ResNet50V2 (TFLite model)
3. **Classification** with confidence score
4. **Recycling Tips** for the detected waste type
""")
st.sidebar.markdown("---")
st.sidebar.header("📈 Model Information")
total_params = 23561152  # Number of trainable parameters in ResNet50V2 base + your dense layers
st.sidebar.info(f"""
**Model Details:**
- Architecture: ResNet50V2 with custom classification head (converted to TFLite)
- Total Parameters: {total_params:,}
- Input Size: 224x224x3
- Classes: {len(CLASS_NAMES)}
""")
st.sidebar.header("🔧 Settings")
show_confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Minimum confidence for reliable predictions"
)
if hasattr(st.session_state, 'confidence'):
    if st.session_state.confidence < show_confidence_threshold:
        st.warning(f"⚠️ Prediction confidence ({st.session_state.confidence*100:.1f}%) is below threshold ({show_confidence_threshold*100:.0f}%). Consider taking another photo.")
