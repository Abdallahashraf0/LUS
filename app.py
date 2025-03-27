# app.py
import streamlit as st
import os
import io
import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# --- Configuration ---
# secrets.toml should contain:
# OPENAI_API_KEY = "your-openai-key"
# HUGGINGFACE_TOKEN = "your-hf-token"

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Streamlit Setup ---
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="wide"
)

# --- Model Loading with Caching ---
@st.cache_resource
def load_multimodal_model():
    """Load the Bio-Medical MultiModal model with caching"""
    # Get Hugging Face token from secrets
    hf_token = st.secrets["hf_token"]
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_id = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"
    
    with st.spinner("Loading Bio-Medical MultiModal model (this may take 2-5 minutes)..."):
        model = AutoModel.from_pretrained(
            model_id,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True
        )
    
    st.success("Multimodal model loaded successfully")
    return model, tokenizer

# Initialize models
model, tokenizer = load_multimodal_model()

# --- Constants ---
QUESTION = (
    "Provide a detailed analysis of this lung ultrasound image. "
    "Include the imaging modality, the organ being examined, detailed observations about the image features, "
    "any abnormalities that you notice (such as B-lines, pleural irregularities, consolidations, etc.), "
    "and any possible treatment recommendations if abnormalities are present."
)

# --- Processing Functions ---
def generate_caption(image):
    """Generate image analysis using the multimodal model"""
    msgs = [{'role': 'user', 'content': [image, QUESTION]}]
    try:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.95,
            stream=False
        )
        return res
    except Exception as e:
        st.error(f"Error during multimodal model chat: {str(e)}")
        return None

def generate_report_with_gpt4o(caption):
    """Generate clinical report using GPT-4o"""
    prompt = (
        f"Based on the following ultrasound image description: '{caption}', "
        "generate a detailed clinical report for a lung ultrasound image with this structure:\n\n"
        "## Findings:\n- B-lines characteristics\n- Pleural abnormalities\n- Consolidations\n- Lung sliding status\n\n"
        "## Impression:\n- Differential diagnosis\n- Confidence assessment\n\n"
        "## Recommendations:\n- Immediate next steps\n- Follow-up schedule\n\n"
        "Use professional medical terminology and maintain clinical accuracy."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a board-certified radiologist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT-4o API Error: {str(e)}")
        return None

# --- Streamlit UI ---
st.title("Lung Ultrasound Analysis System 🩺")
st.markdown("""
    Upload a lung ultrasound image for AI-powered analysis.  
    This system combines multimodal AI with clinical expertise.
""")

uploaded_file = st.file_uploader(
    "Choose an ultrasound image (JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Patient Scan")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True, caption="Uploaded Ultrasound Image")
    
    with col2:
        with st.status("Analyzing image features...", expanded=True) as status:
            caption = generate_caption(image)
            
            if caption:
                status.update(label="Analysis Complete", state="complete", expanded=False)
                st.subheader("Preliminary Findings")
                st.write(caption)
                
                with st.spinner("Generating clinical report..."):
                    report = generate_report_with_gpt4o(caption)
                
                if report:
                    st.subheader("Clinical Report")
                    st.markdown(f"\n{report}\n")

st.markdown("---")
st.markdown("""
    **Clinical Disclaimer**:  
    This AI analysis is for informational purposes only. Always consult a qualified healthcare professional for medical diagnosis and treatment decisions.
""")
