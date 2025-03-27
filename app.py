import os
import torch
import openai
import nest_asyncio
from PIL import Image
import streamlit as st
from transformers import AutoModel, AutoTokenizer

# Fix async issues
nest_asyncio.apply()

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["hf_token"]

st.title("AI-Powered Lung Ultrasound Analysis")
st.write("Loading Bio-Medical MultiModal model... (this may take several minutes)")

# --- Model Loading Fixes ---
model_id = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"

@st.cache_resource(show_spinner=False)
def load_model():
    # Patch tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=hf_token
    )
    
    # Critical tokenizer patch
    if not hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer = tokenizer
    
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )
    return model, tokenizer

# Load with progress
with st.spinner("Initializing medical AI model..."):
    model, tokenizer = load_model()
st.success("Model loaded successfully!")

# --- Constants ---
QUESTION = (
    "Provide a detailed analysis of this lung ultrasound image. "
    "Include imaging modality, organ examined, detailed observations of image features, "
    "any abnormalities (B-lines, pleural irregularities, consolidations), "
    "and possible treatment recommendations."
)

# --- Core Functions ---
def generate_caption(image):
    """Generate analysis using the multimodal model"""
    try:
        msgs = [{'role': 'user', 'content': [image, QUESTION]}]
        return model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.95,
            stream=False
        )
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def generate_report_with_gpt4o(caption):
    """Generate clinical report using GPT-4o"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a medical AI assistant expert. Generate a structured report with Findings, Impression, and Recommendations."
            }, {
                "role": "user",
                "content": f"Ultrasound analysis: {caption}\n\nGenerate detailed clinical report with:\n"
                           "- Quantitative B-line counts\n- Pleural line assessment\n- Consolidation characteristics\n"
                           "- Differential diagnosis confidence levels\n- Evidence-based recommendations"
            }],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Report error: {str(e)}")
        return None

# --- UI Flow ---
uploaded_file = st.file_uploader("Upload lung ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
    
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing medical image..."):
            caption = generate_caption(image)
        
        if caption:
            with st.spinner("Generating clinical report..."):
                report = generate_report_with_gpt4o(caption)
            
            st.divider()
            st.subheader("Clinical Analysis Report")
            st.markdown(f"**AI Findings**\n{caption}")
            st.markdown(f"**Clinical Report**\n{report}")
        else:
            st.error("Analysis failed - please try a different image")

# Add system requirements
st.sidebar.markdown("**System Requirements**\n- CUDA 12.1+\n- PyTorch 2.1.2+\n- Python 3.10+")
