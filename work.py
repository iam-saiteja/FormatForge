import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from google import genai
import os
import time
from google.genai import types

# Set page config
st.set_page_config(
    page_title="FormatForge",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = {}

# Default API key empty; user should enter in sidebar
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Supported shot angles (max 4)
ANGLES = [
    "front view",
    "left 45 degree angle",
    "right 45 degree angle",
    "top-angled view"
]

def parse_size(size_str):
    # Expect formats like '1000x1000px' or '1080x1080px'
    try:
        parts = size_str.lower().replace('px', '').split('x')
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return None, None

def resize_image_file(path, width, height):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((width, height), Image.LANCZOS)
        img.save(path, format='JPEG')
        return True
    except Exception:
        return False

# Platform specifications
PLATFORMS = {
    "Amazon": {
        "description": "E-commerce product listing",
        "requirements": "Pure white background, no props, product fills 85% of frame",
        "aspect_ratio": "1:1",
        "size": "1000x1000px"
    },
    "Flipkart": {
        "description": "E-commerce product listing",
        "requirements": "White background, no watermarks, clear product view",
        "aspect_ratio": "1:1",
        "size": "1000x1000px"
    },
    "Zomato": {
        "description": "Food delivery listing",
        "requirements": "Appetizing food presentation, enhanced colors",
        "aspect_ratio": "1:1",
        "size": "1080x1080px"
    },
    "Swiggy": {
        "description": "Food delivery listing",
        "requirements": "Appetizing food presentation, vibrant colors",
        "aspect_ratio": "1:1",
        "size": "1080x1080px"
    },
    "Instagram Post": {
        "description": "Social media post",
        "requirements": "Engaging composition, high aesthetic value",
        "aspect_ratio": "1:1 or 4:5",
        "size": "1080x1080px"
    },
    "Instagram Story": {
        "description": "Social media story",
        "requirements": "Vertical format, space for text/graphics",
        "aspect_ratio": "9:16",
        "size": "1080x1920px"
    },
    "OLX": {
        "description": "Classifieds listing",
        "requirements": "Clean background, no personal info, clear product view",
        "aspect_ratio": "4:3 or 1:1",
        "size": "800x800px"
    },
    "Spotify Canvas": {
        "description": "Music visualizer",
        "requirements": "Vertical format, music-themed, visually striking",
        "aspect_ratio": "9:16",
        "size": "1080x1920px"
    }
}

# Prompts for Gemini
PROMPTS = {
    "Amazon": "I am providing an image of a product. You need to edit this image to meet Amazon's product listing requirements. The final image must have a pure white background (RGB 255, 255, 255), the product must fill 85% of the frame, and it should be in a 1:1 aspect ratio. Remove any props, logos, or watermarks. Your output should only be the edited image.",
    "Flipkart": "I am providing an image of a product. You need to edit this image to meet Flipkart's product listing requirements. The final image must have a white background, be centered, and have a 1:1 aspect ratio. Remove any watermarks or text. Your output should only be the edited image.",
    "Zomato": "I am providing an image of a food dish. You need to edit this image to make it suitable for a Zomato listing. Enhance the colors to make the food look appetizing, increase saturation and contrast, and crop it to a 1:1 square aspect ratio. Your output should only be the edited image.",
    "Swiggy": "I am providing an image of a food dish. You need to edit this image for a Swiggy listing. Make the food look delicious and vibrant, and ensure the image is clear and inviting with a 1:1 aspect ratio. Your output should only be the edited image.",
    "Instagram Post": "I am providing an image. You need to adapt it for an Instagram post. Enhance the colors to make it engaging and stylish, and set the aspect ratio to 1:1. Your output should only be the edited image.",
    "Instagram Story": "I am providing an image. You need to reformat it for an Instagram Story. Change the aspect ratio to 9:16 (vertical), ensure the main subject is centered, and leave some space at the top and bottom if possible. Your output should only be the edited image.",
    "OLX": "I am providing an image of a product for an OLX classifieds listing. You need to edit it to make the product clearly visible. Remove any background clutter and set the background to a neutral color like gray or white. Your output should only be the edited image.",
    "Spotify Canvas": "I am providing an image. You need to create a vertical image from it for a Spotify Canvas. The aspect ratio must be 9:16. The final image should be visually striking and music-related. If it's an album cover, adapt it to fit vertically while keeping the key elements. Your output should only be the edited image."
}

# Function to encode image
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def validate_and_fix_b64(b64_str):
    """Try to validate that b64_str decodes to a valid image.
    If needed, try common fixes (strip data URI, attempt double-decode).
    Returns a valid base64 string or None if unfixable.
    """
    if not b64_str:
        return None

    def try_open(b64candidate):
        try:
            raw = base64.b64decode(b64candidate)
        except Exception:
            return False
        try:
            img = Image.open(BytesIO(raw))
            img.verify()
            return True
        except Exception:
            return False

    # Direct attempt
    if try_open(b64_str):
        return b64_str

    # Strip common data URI prefix
    if b64_str.startswith("data:image"):
        try:
            b64_clean = b64_str.split(",", 1)[1]
        except Exception:
            b64_clean = b64_str
        if try_open(b64_clean):
            return b64_clean

    # If the decoded bytes are ASCII-looking, maybe it was double-encoded: decode once and try again
    try:
        decoded_once = base64.b64decode(b64_str)
        # If decoded_once looks like base64 text, try decode again
        try:
            decoded_once_text = decoded_once.decode('utf-8', errors='ignore')
            # Heuristic: contains typical base64 chars and padding
            if all(c.isalnum() or c in '+/=' for c in decoded_once_text.strip()[:10]):
                if try_open(decoded_once_text):
                    return decoded_once_text
                # try decoding bytes again
                try:
                    if try_open(base64.b64encode(decoded_once).decode('utf-8')):
                        return base64.b64encode(decoded_once).decode('utf-8')
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    return None

# Function to call Gemini API
def call_gemini_api(image_b64, prompt, platform):
    # Create a client using the provided API key (matches magic.py example)
    client = genai.Client(api_key=st.session_state.api_key)

    # Decode image to a PIL Image and pass it directly to the SDK (preferred)
    try:
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to decode uploaded image: {e}")
        return None

    # Build contents: text instruction first, then the PIL Image
    text_instruction = f"Edit this image to meet the following requirements: {prompt}. Output only the edited image."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[text_instruction, pil_img],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )
    except Exception as e:
        st.error(f"API call failed for {platform}: {e}")
        return None

    # Parse response: support both parts/as_image() and candidates/inline_data formats
    try:
        # Preferred: response.parts (newer SDK) with part.as_image()
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                    # part.as_image() may return different types depending on SDK/runtime.
                    try:
                        img_obj = part.as_image()
                    except Exception:
                        img_obj = None
                    if img_obj:
                        # Normalize to raw JPEG bytes regardless of object type
                        try:
                            # If raw bytes
                            if isinstance(img_obj, (bytes, bytearray)):
                                img_bytes = bytes(img_obj)
                            # IPython.display.Image-like object with .data
                            elif hasattr(img_obj, 'data'):
                                img_bytes = img_obj.data
                            else:
                                # Assume PIL Image or similar with save(); try saving to JPEG
                                tmp_buf = BytesIO()
                                try:
                                    img_obj.save(tmp_buf, format='JPEG')
                                except TypeError:
                                    # Some objects' save may not accept format kwarg
                                    img_obj.save(tmp_buf)
                                img_bytes = tmp_buf.getvalue()

                            return base64.b64encode(img_bytes).decode('utf-8')
                        except Exception:
                            # if normalization fails, skip to try other parts/candidates
                            continue

        # Fallback: older candidate style with inline_data
        if hasattr(response, 'candidates') and response.candidates:
            image_parts = [
                part.inline_data.data
                for part in response.candidates[0].content.parts
                if hasattr(part, 'inline_data') and part.inline_data
            ]
            if image_parts:
                return base64.b64encode(image_parts[0]).decode('utf-8')

    except Exception as e:
        try:
            resp_text = getattr(response, 'text', str(response))
        except Exception:
            resp_text = str(response)
        st.error(f"Image generation failed for {platform}. Debug: {e}. Response: {resp_text}")
        return None

    st.error(f"Image generation returned no image for {platform}.")
    return None

# Function to generate audio feedback (simulated)
def generate_audio_feedback(platform, success):
    # In a real implementation, you'd use ElevenLabs API
    feedback_text = f"Image for {platform} generated successfully. All requirements met." if success else f"Image for {platform} generation failed."
    return feedback_text

# UI Layout
st.title("🛠️ FormatForge")
st.markdown("### One Asset, Perfectly Formatted for Every Platform")

# Sidebar for platform info
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Gemini API key", value=st.session_state.api_key, type="password")
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.header("Generation")
    num_images = st.slider("How many images per platform?", 1, 4, 1)
    st.write("Each generated image will attempt a different angle up to the selected number.")
    st.markdown("**Platform-specific extras**")
    spotify_extra = st.text_input("Spotify Canvas: any specific elements to include?", value="")
    olx_extra = st.text_input("OLX: any specific details to show (condition, accessories)?", value="")

    st.markdown("---")
    st.header("Platform Specifications")
    selected_platform_info = st.selectbox("View requirements for:", list(PLATFORMS.keys()))
    st.write(f"**{selected_platform_info}**")
    st.write(f"Description: {PLATFORMS[selected_platform_info]['description']}")
    st.write(f"Requirements: {PLATFORMS[selected_platform_info]['requirements']}")
    st.write(f"Aspect Ratio: {PLATFORMS[selected_platform_info]['aspect_ratio']}")
    st.write(f"Recommended Size: {PLATFORMS[selected_platform_info]['size']}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Your Image")
    uploaded_files = st.file_uploader("Choose up to 4 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        # Limit to 4
        if len(uploaded_files) > 4:
            st.warning("Please upload at most 4 images. Only the first 4 will be used.")
            uploaded_files = uploaded_files[:4]

        # Display thumbnails and let user choose whether to generate for all
        cols = st.columns(len(uploaded_files))
        for i, uf in enumerate(uploaded_files):
            try:
                thumb = Image.open(uf)
                cols[i].image(thumb, use_column_width=True)
                cols[i].write(uf.name)
            except Exception:
                cols[i].write(uf.name)

        generate_for_all = st.checkbox("Generate for all uploaded images?", value=False)
        selected_index = 0
        if not generate_for_all:
            names = [f.name for f in uploaded_files]
            selected_name = st.selectbox("Select which uploaded image to use:", names)
            selected_index = names.index(selected_name)

        if generate_for_all:
            images_to_process = uploaded_files
        else:
            images_to_process = [uploaded_files[selected_index]]

        # For preview, pick first
        image = Image.open(images_to_process[0])
        st.image(image, caption="Original Image", use_column_width=True)
        
        st.header("Select Platforms")
        selected_platforms = []
        for platform in PLATFORMS.keys():
            if st.checkbox(platform, key=platform):
                selected_platforms.append(platform)
        
        if st.button("Generate Formatted Images", disabled=len(selected_platforms) == 0 or not st.session_state.api_key):
            if len(selected_platforms) == 0:
                st.error("Please select at least one platform.")
            else:
                with st.spinner("Generating formatted images..."):
                    # Clear all previously generated images to start fresh for this generation.
                    try:
                        gen_dir = os.path.join(os.getcwd(), 'generated')
                        if os.path.exists(gen_dir):
                            for fname in os.listdir(gen_dir):
                                fpath = os.path.join(gen_dir, fname)
                                try:
                                    if os.path.isfile(fpath):
                                        os.remove(fpath)
                                except Exception:
                                    pass
                        # Reset session state container so UI doesn't keep old entries
                        st.session_state.generated_images = {}
                    except Exception:
                        # If cleanup fails, continue but warn
                        st.warning("Failed to fully clear previous generated images, continuing anyway.")
                    # Process each selected source image
                    for src_idx, src_file in enumerate(images_to_process):
                        try:
                            src_img = Image.open(src_file).convert('RGB')
                        except Exception:
                            st.error(f"Failed to open uploaded image: {getattr(src_file,'name',str(src_idx))}")
                            continue

                        img_b64 = encode_image(src_img)

                        # Generate images for each platform
                        for platform in selected_platforms:
                            with st.status(f"Processing {platform}...", expanded=True) as status:
                                try:
                                    # Generate multiple angles up to num_images
                                    platform_size = PLATFORMS[platform]['size']
                                    w, h = parse_size(platform_size)
                                    generated_list = []
                                    for i in range(num_images):
                                        angle = ANGLES[i]
                                        # Append any platform extras
                                        extra = ""
                                        if platform == 'Spotify Canvas' and spotify_extra:
                                            extra = f" Include: {spotify_extra}."
                                        if platform == 'OLX' and olx_extra:
                                            extra = f" Include: {olx_extra}."

                                        angle_prompt = f"{PROMPTS[platform]} Generate the image from a {angle}.{extra}"
                                        result_b64 = call_gemini_api(img_b64, angle_prompt, platform)

                                        if not result_b64:
                                            continue

                                        fixed_b64 = validate_and_fix_b64(result_b64)
                                        if not fixed_b64:
                                            continue

                                        # Decode and save to a physical file
                                        img_bytes = base64.b64decode(fixed_b64)
                                        os.makedirs(os.path.join(os.getcwd(), 'generated'), exist_ok=True)
                                        fname = f"formatforge_{platform.lower().replace(' ', '_')}_{i}_{int(time.time())}.jpg"
                                        fpath = os.path.join(os.getcwd(), 'generated', fname)
                                        with open(fpath, 'wb') as f:
                                            f.write(img_bytes)

                                        # Resize to platform size if parse succeeded
                                        if w and h:
                                            resize_image_file(fpath, w, h)

                                        generated_list.append({"b64": fixed_b64, "path": fpath, "angle": angle, "modified": False})

                                    if generated_list:
                                        st.session_state.generated_images[platform] = generated_list
                                        status.update(label=f"{platform}: Complete", state="complete")
                                        st.write(generate_audio_feedback(platform, True))
                                    else:
                                        status.update(label=f"{platform}: Error", state="error")
                                except Exception as e:
                                    status.update(label=f"{platform}: Error", state="error")
                                    st.error(f"Error generating image for {platform}: {str(e)}")

with col2:
    st.header("Generated Images")
    
    if st.session_state.generated_images:
        for platform, data in st.session_state.generated_images.items():
            st.subheader(platform)

            # If multiple generated images (list), show each
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    angle = item.get('angle', f'image_{idx+1}')
                    st.write(f"Angle: {angle}")
                    fpath = item.get('path')
                    b64str = item.get('b64')
                    img = None
                    if fpath and os.path.exists(fpath):
                        try:
                            img = Image.open(fpath)
                        except Exception:
                            img = None
                    if img is None and b64str:
                        try:
                            img = Image.open(BytesIO(base64.b64decode(b64str)))
                        except Exception:
                            img = None

                    if img:
                        st.image(img, use_column_width=True)
                        # Download button per item
                        buf = BytesIO()
                        try:
                            img.save(buf, format='JPEG')
                            byte_im = buf.getvalue()
                            st.download_button(
                                label=f"Download {platform} ({idx+1})",
                                data=byte_im,
                                file_name=f"formatforge_{platform.lower().replace(' ', '_')}_{idx+1}.jpg",
                                mime='image/jpeg',
                                key=f"download_{platform}_{idx}"
                            )
                        except Exception as e:
                            st.error(f"Failed to prepare download for {platform} item {idx+1}: {e}")

                        # Modify UI per item with busy/pending flags
                        modify_key = f"modify_input_{platform}_{idx}"
                        modify_text = st.text_input("Modify / Chat about this image:", key=modify_key)
                        busy_key = f"modify_busy_{platform}_{idx}"
                        pending_key = f"modify_pending_{platform}_{idx}"

                        # Ensure flags exist
                        if busy_key not in st.session_state:
                            st.session_state[busy_key] = False
                        if pending_key not in st.session_state:
                            st.session_state[pending_key] = None

                        # Button is disabled when busy
                        is_disabled = bool(st.session_state.get(busy_key, False))
                        if is_disabled:
                            st.info("Modification in progress...")

                        if st.button("Apply modification", key=f"modify_btn_{platform}_{idx}", disabled=is_disabled):
                            if not modify_text:
                                st.error("Enter modification instructions first.")
                            else:
                                # Queue the modification and trigger a rerun so UI can show disabled state
                                st.session_state[pending_key] = modify_text
                                st.session_state[busy_key] = True
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    try:
                                        st.query_params = {"_mod": str(int(time.time()))}
                                    except Exception:
                                        # If rerun attempt fails, we'll continue in this run; show spinner below
                                        pass

                        # If a modification was queued for this item, perform it now (this will run on rerun)
                        if st.session_state.get(pending_key) and st.session_state.get(busy_key):
                            queued_text = st.session_state.get(pending_key)
                            # Show spinner and status while processing
                            with st.spinner(f"Applying modification to {platform} ({angle})..."):
                                try:
                                    pil_src = Image.open(fpath).convert('RGB')
                                    src_b64 = encode_image(pil_src)
                                    edit_prompt = f"Edit the provided image: {queued_text}. Output only the edited image."
                                    result_b64 = call_gemini_api(src_b64, edit_prompt, platform)
                                    fixed = validate_and_fix_b64(result_b64)
                                    if not fixed:
                                        st.error("Model returned no valid image for modification.")
                                    else:
                                        img_bytes = base64.b64decode(fixed)
                                        # Overwrite same file so item stays in-place
                                        with open(fpath, 'wb') as f:
                                            f.write(img_bytes)
                                        pw, ph = parse_size(PLATFORMS[platform]['size'])
                                        if pw and ph:
                                            resize_image_file(fpath, pw, ph)
                                        # Update session state explicitly and mark as modified
                                        updated_item = {**data[idx], 'b64': fixed, 'modified': True}
                                        st.session_state.generated_images[platform][idx] = updated_item
                                        # Clear pending and clear busy
                                        st.session_state[pending_key] = None
                                        st.session_state[busy_key] = False
                                        st.success("Image modified and updated in place.")
                                        # Immediately reload and display the updated image so UI reflects the change
                                        try:
                                            refreshed = Image.open(fpath)
                                            st.image(refreshed, use_column_width=True)
                                        except Exception:
                                            pass
                                        # Try to refresh once more so the UI shows the new image
                                        try:
                                            st.experimental_rerun()
                                        except Exception:
                                            try:
                                                    st.query_params = {"_refresh": str(int(time.time()))}
                                            except Exception:
                                                st.session_state._last_modify_ts = int(time.time())
                                except Exception as e:
                                    st.session_state[pending_key] = None
                                    st.session_state[busy_key] = False
                                    st.error(f"Modification failed: {e}")
                    else:
                        st.error(f"Could not open generated image for {platform} item {idx+1}. Path: {fpath}")

            else:
                # Single generated image (dict or base64 string)
                img = None
                if isinstance(data, dict):
                    fpath = data.get('path')
                    b64str = data.get('b64')
                    if fpath and os.path.exists(fpath):
                        try:
                            img = Image.open(fpath)
                        except Exception:
                            img = None
                    if img is None and b64str:
                        try:
                            img = Image.open(BytesIO(base64.b64decode(b64str)))
                        except Exception:
                            img = None
                else:
                    try:
                        img = Image.open(BytesIO(base64.b64decode(data)))
                    except Exception:
                        img = None

                if img:
                    st.image(img, use_column_width=True)
                    buf = BytesIO()
                    try:
                        img.save(buf, format='JPEG')
                        byte_im = buf.getvalue()
                        st.download_button(
                            label=f"Download {platform} Image",
                            data=byte_im,
                            file_name=f"formatforge_{platform.lower().replace(' ', '_')}.jpg",
                            mime="image/jpeg",
                            key=f"download_{platform}"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare download for {platform}: {e}")
                else:
                    # provide path for debugging if available
                    pth = data.get('path') if isinstance(data, dict) else None
                    st.error(f"Could not open generated image for {platform}. Path: {pth}")
    else:
        st.info("Upload an image and select platforms to generate formatted versions.")

# Footer
st.markdown("---")
st.markdown("""
### How It Works
1. **Upload** your original image
2. **Select** the platforms you want to format for
3. **Generate** optimized images using Gemini 1.5 Flash
4. **Download** your perfectly formatted assets

### Powered By
- **Gemini 1.5 Flash** for intelligent image transformation
- **Fal AI** for batch processing capabilities (simulation)
- **ElevenLabs** for voice feedback and validation (simulation)
""")