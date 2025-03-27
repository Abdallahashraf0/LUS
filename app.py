import os
import torch
import openai
import nest_asyncio
from PIL import Image
import streamlit as st
from transformers import AutoModel, AutoTokenizer

# Fix async event loop conflicts
nest_asyncio.apply()

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["hf_token"]

st.title("Medical Imaging Analysis System")
st.write("Initializing Bio-Medical MultiModal Model...")

# --- Model Configuration ---
model_id = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"
QUESTION = 'Give the modality, organ, analysis, abnormalities (if any), treatment (if abnormalities are present)?'

@st.cache_resource(show_spinner=False)
def load_model_components():
    # Load tokenizer with remote code support
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=hf_token
    )
    
    # Load model with Flash Attention
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=hf_token,
        # attn_implementation="flash_attention_2"
    )
    
    return model, tokenizer

# Load components with progress
with st.spinner("Loading medical AI engine (this may take 3-5 minutes)..."):
    model, tokenizer = load_model_components()
st.success("System ready for analysis!")

# --- Core Analysis Functions ---
def analyze_medical_image(image):
    """Process image using the biomedical model"""
    try:
        msgs = [{'role': 'user', 'content': [image, QUESTION]}]
        response_stream = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.95,
            stream=True  # Use streaming as per documentation
        )
        
        # Stream the response incrementally
        full_response = ""
        placeholder = st.empty()
        for chunk in response_stream:
            full_response += chunk
            placeholder.markdown(f"**AI Analysis:**\n{full_response}▌")
        
        placeholder.markdown(f"**AI Analysis:**\n{full_response}")
        return full_response
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# --- Clinical Report Generation ---
def generate_clinical_report(analysis):
    """Generate structured report using GPT-4o"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "Transform this medical analysis into a structured clinical report:"
            }, {
                "role": "user",
                "content": f"Analysis: {analysis}\n\nFormat:\n"
                           "1. Modality & Organ\n2. Key Findings\n"
                           "3. Abnormalities\n4. Treatment Recommendations\n"
                           "Use medical terminology from latest guidelines."
            }],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return None

# --- UI Flow ---
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Analyze Image", type="primary"):
        analysis = analyze_medical_image(image)
        
        if analysis:
            with st.spinner("Generating clinical report..."):
                report = generate_clinical_report(analysis)
            
            st.divider()
            st.subheader("Clinical Report")
            st.markdown(f"**Analysis Summary**\n{analysis}")
            st.markdown(f"**Structured Report**\n{report}")
            
            # Add export capability
            st.download_button(
                label="Download Full Report",
                data=f"Analysis Summary:\n{analysis}\n\nClinical Report:\n{report}",
                file_name="medical_analysis_report.txt",
                mime="text/plain"
            )

# --- Sidebar Information ---
st.sidebar.markdown("""
**System Requirements**
- PyTorch 2.1.2+
- CUDA 12.1+ (NVIDIA GPUs)
- 16GB+ VRAM recommended
- Python 3.10+

**Model Limitations**
- For non-commercial use only
- Verify critical findings with clinical experts
- May contain inherent training biases
""")
