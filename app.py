import os
import io
import torch
import openai
from PIL import Image
import streamlit as st
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Set API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["hf_token"]

st.title("AI-Powered Lung Ultrasound Analysis")
st.write("Loading Bio-Medical MultiModal model... (this may take a few minutes)")

# Configure BitsAndBytes for 4-bit quantization (matches Colab config)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Define model ID and load model/tokenizer
model_id = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"
model = AutoModel.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # Enabled quantization
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_auth_token=hf_token,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_auth_token=hf_token
)

st.success("Multimodal model loaded successfully!\n")

# --- Define Task Variables ---
QUESTION = (
    "Provide a detailed analysis of this lung ultrasound image. "
    "Include the imaging modality, the organ being examined, detailed observations about the image features, "
    "any abnormalities that you notice (such as B-lines, pleural irregularities, consolidations, etc.), "
    "and any possible treatment recommendations if abnormalities are present."
)

def generate_caption(image):
    """Generates analysis using the multimodal model"""
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
        st.error(f"Error during analysis: {e}")
        return None

def generate_report_with_gpt4o(caption):
    """Generates clinical report using GPT-4o"""
    prompt = (
        f"Based on: '{caption}', generate a detailed clinical report with:\n"
        "Findings:\n- B-lines\n- Pleural abnormalities\n- Consolidations\n- Lung sliding\n\n"
        "Impression:\n- Differential diagnosis\n- Confidence level\n\n"
        "Recommendations:\n- Next steps\n- Follow-up timing\n\n"
        "Use medical terminology and be detailed."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "system", 
                "content": "You are a medical AI assistant expert."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None

# --- File Upload & Processing ---
uploaded_file = st.file_uploader("Upload lung ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
    
    caption = generate_caption(image)
    if caption:
        report = generate_report_with_gpt4o(caption)
        st.markdown("---")
        st.subheader("Analysis Results")
        st.markdown(f"**AI Analysis:**\n{caption}")
        st.markdown(f"**Clinical Report:**\n{report}")
    else:
        st.error("Failed to generate analysis")
