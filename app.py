import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="♻️ Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .result-box h2 {
        margin: 0;
        font-size: 2.5rem;
    }
    .result-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("♻️ Smart Waste Classifier")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Upload an image and let AI identify the waste type for proper recycling!
        </p>
    </div>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        # Load your trained model
        model = tf.keras.models.load_model("Waste_Classification_Model.keras")
        return model
    except:
        st.error("⚠️ Model not found! Please train and save the model first.")
        return None

# Get class names from the model
@st.cache_resource
def get_class_names():
    train_directory = Path(__file__).parent / "train"
    return [folder.name.title() for folder in sorted(train_directory.iterdir()) if folder.is_dir()]

# Load model
model = load_model()
class_names = get_class_names()
if model is not None and model.output_shape[-1] != len(class_names):
    st.error("The model output does not match the training categories.")
    model = None

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
        1. 📸 Upload an image of waste
        2. 🔍 AI analyzes the image
        3. ♻️ Get instant classification results
        
        **Supported categories:**
        - 📦 Cardboard
        - 🫙 Glass
        - 🔩 Metal
        - 📄 Paper
        - 🧴 Plastic
        - 🗑️ Trash
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Accuracy Tips")
    st.markdown("""
        - Use clear, well-lit images
        - Center the waste item
        - Avoid cluttered backgrounds
        - Use high-resolution images
    """)
    
    st.markdown("---")
    st.markdown("### 🌍 Why Recycle?")
    st.markdown("""
        - Reduces landfill waste
        - Conserves natural resources
        - Saves energy
        - Reduces pollution
    """)

# Main upload area
uploaded_file = st.file_uploader(
    "📤 Choose a waste image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload an image of waste for classification"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True, output_format='auto')
    
    # Process image and make prediction
    if model is not None:
        with st.spinner("🧠 AI is analyzing the image..."):
            # Preprocess image using the dimensions expected by the loaded model.
            _, img_height, img_width, _ = model.input_shape
            if img_height is None or img_width is None:
                st.error("The model does not define fixed image dimensions.")
                st.stop()
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize only; the saved model already includes a Rescaling layer.
            image = image.resize((img_width, img_height))
            img_array = np.array(image)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = model.predict(img_batch)
            score = tf.nn.softmax(predictions[0])
            
            # Get the predicted class
            predicted_class = class_names[np.argmax(score)]
            confidence = np.max(score) * 100
            
            # Display results
            st.markdown("---")
            
            # Result box
            st.markdown(f"""
                <div class="result-box">
                    <p>🎯 Prediction Result</p>
                    <h2>{predicted_class}</h2>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display confidence meter
            st.markdown("### 📊 Confidence Level")
            progress_color = "green" if confidence > 70 else "orange" if confidence > 50 else "red"
            st.markdown(f"""
                <div style="width:100%; background:#e0e0e0; border-radius:10px; height:20px;">
                    <div style="width:{confidence}%; background:{progress_color}; 
                        border-radius:10px; height:20px; transition: width 1s ease;">
                    </div>
                </div>
                <p style="text-align:right; margin:0; color:{progress_color};">
                    {confidence:.1f}%
                </p>
            """, unsafe_allow_html=True)
            
            # Show probabilities for all classes
            with st.expander("🔍 View all probabilities", expanded=False):
                st.markdown("#### Class Probabilities")
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(class_names, score.numpy() * 100, color='skyblue')
                ax.set_xlabel('Probability (%)')
                ax.set_title('Classification Probabilities')
                
                # Highlight the highest probability
                max_idx = np.argmax(score)
                bars[max_idx].set_color('#4CAF50')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, score.numpy() * 100)):
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', va='center', fontsize=10)
                
                st.pyplot(fig)
                
                # Display numerical values
                st.markdown("#### Detailed Probabilities")
                cols = st.columns(3)
                for idx, (name, prob) in enumerate(zip(class_names, score.numpy() * 100)):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        st.metric(label=name, value=f"{prob:.1f}%")
            
            # Suggestions based on classification
            st.markdown("### 💡 Recycling Tips")
            tips = {
                'Cardboard': "♻️ Flatten boxes, remove tape, and place in paper recycling. Keep dry!",
                'Glass': "🫙 Rinse containers, remove lids, and place in glass recycling. Do not break!",
                'Metal': "🔩 Clean cans, remove labels, and place in metal recycling. Aluminum and steel are valuable!",
                'Paper': "📄 Remove plastic windows, keep clean, and place in paper recycling. Shred sensitive documents!",
                'Plastic': "🧴 Check recycling numbers, rinse containers, and place in plastic recycling. Crush to save space!",
                'Trash': "🗑️ Dispose in general waste. Consider if any parts can be recycled separately!"
            }
            
            tip = tips.get(predicted_class, "♻️ Please check local recycling guidelines.")
            st.info(tip)
            
            # Fact about the waste type
            facts = {
                'Cardboard': "📊 Fact: Recycling one ton of cardboard saves 17 trees and 7,000 gallons of water!",
                'Glass': "📊 Fact: Glass can be recycled infinitely without loss of quality!",
                'Metal': "📊 Fact: Recycling aluminum saves 95% of the energy needed to make new aluminum!",
                'Paper': "📊 Fact: Paper can be recycled 5-7 times before fibers become too short!",
                'Plastic': "📊 Fact: Only 9% of all plastic ever produced has been recycled!",
                'Trash': "📊 Fact: The average American generates 4.5 pounds of trash daily!"
            }
            
            fact = facts.get(predicted_class, "🌍 Every item you recycle makes a difference!")
            st.success(fact)
            
    else:
        st.error("⚠️ Please train and save your model first (Waste_Classification_Model.keras)")
else:
    # Show example images when no file is uploaded
    st.markdown("### 📷 Try with sample images")
    st.info("👆 Upload an image above to get started!")
    
    # Quick tips
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 10px;'>
                📦<br>
                Cardboard
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 10px;'>
                🫙<br>
                Glass
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 10px;'>
                🔩<br>
                Metal
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; margin-top: 2rem; font-size: 0.9rem; color: #999;'>
            🤖 Powered by TensorFlow and Deep Learning
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.8rem;'>
        Made with ❤️ for a greener planet • Waste Classification AI
    </div>
""", unsafe_allow_html=True)