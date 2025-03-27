import os
import io
import torch
import openai
from PIL import Image
import streamlit as st
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Set your OpenAI API key (or set it as an environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
hf_token = st.secrets["hf_token"]

st.title("AI-Powered Lung Ultrasound Analysis")
st.write("Loading Bio-Medical MultiModal model... (this may take a few minutes)")

# Uncomment and configure BitsAndBytes if you wish to use quantization.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )

# Define model ID and load model/tokenizer with the token
model_id = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"
model = AutoModel.from_pretrained(
    model_id,
    # quantization_config=bnb_config,  # Quantization disabled
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
# Patch the tokenizer if necessary (this avoids missing attribute issues)
if not hasattr(tokenizer, "tokenizer"):
    tokenizer.tokenizer = tokenizer

st.success("Multimodal model loaded successfully.\n")

# Define a detailed question that aligns with the GPT-4o prompt
QUESTION = (
    "Provide a detailed analysis of this lung ultrasound image. "
    "Include the imaging modality, the organ being examined, detailed observations about the image features, "
    "any abnormalities that you notice (such as B-lines, pleural irregularities, consolidations, etc.), "
    "and any possible treatment recommendations if abnormalities are present."
)

def generate_caption(image):
    """
    Uses the model.chat method to generate a caption/answer from the image.
    The message content now contains only the detailed question.
    The image is provided via the image parameter.
    """
    msgs = [{'role': 'user', 'content': [QUESTION]}]
    try:
        # Pass the image only via the keyword parameter; the message contains only the question.
        res = model.chat(image=image, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.95, stream=False)
        st.write("**Generated Caption:**", res)
        return res
    except Exception as e:
        st.error(f"Error during multimodal model chat: {e}")
        return None

def generate_report_with_gpt4o(caption):
    """
    Uses GPT-4o to generate a detailed clinical report based on the provided caption.
    """
    prompt = (
        f"Based on the following ultrasound image description: '{caption}', "
        "generate a detailed clinical report for a lung ultrasound image with the following structure:\n\n"
        "Findings:\n"
        "- Describe B-lines (include counts if possible)\n"
        "- Note pleural abnormalities\n"
        "- Identify any consolidations\n"
        "- Mention lung sliding presence/absence\n\n"
        "Impression:\n"
        "- Provide a differential diagnosis\n"
        "- Rate confidence level\n\n"
        "Recommendations:\n"
        "- Suggest next clinical steps\n"
        "- Recommend follow-up timing\n\n"
        "Use clear medical terminology and provide a coherent, detailed report."
    )
    messages = [
        {"role": "system", "content": "You are a medical AI assistant expert."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            top_p=1
        )
        report = response.choices[0].message.content.strip()
        st.write("**Generated Clinical Report:**", report)
        return report
    except Exception as e:
        st.error(f"Error during GPT-4o call: {e}")
        return None

# File uploader for image input
uploaded_file = st.file_uploader("Choose a lung ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    caption = generate_caption(image)
    if caption:
        report = generate_report_with_gpt4o(caption)
        st.markdown("---")
        st.subheader("Final Results")
        st.write("**Caption:**", caption)
        st.write("**Clinical Report:**", report)
    else:
        st.error("Caption generation failed.")
