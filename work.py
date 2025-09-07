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
# Optional global extra instructions field (user may leave empty)
if 'extra_instructions' not in st.session_state:
    st.session_state.extra_instructions = ""

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

# Enhanced Platform Specifications (Pixel-Perfect, Scaling-Safe)
PLATFORMS = {
    "Amazon": {
        "description": "Amazon e-commerce product listing (main image)",
        "requirements": {
            "background": "Pure white (RGB 255,255,255 / HEX #FFFFFF)",
            "frame_fill": "Product must occupy 85–90% of the frame",
            "prohibited": ["props", "logos", "watermarks", "text", "inset images"],
            "focus": "Sharp, professional, minimal shadows",
            "color": "True-to-life, accurate product color",
            "aspect_ratio": "1:1",
            "size": "2000x2000px (minimum 1000x1000px)",
            "format": "JPEG or TIFF",
            "color_space": "sRGB"
        },
        "notes": "Resize by scaling/reducing proportionally. Do not crop subject; extend white background if necessary."
    },
    "Flipkart": {
        "description": "Flipkart e-commerce product listing",
        "requirements": {
            "background": "Clean white (RGB ~245,245,245)",
            "frame_fill": "Product should occupy at least 80% of the frame",
            "prohibited": ["watermarks", "logos", "promotional text"],
            "focus": "Clear visibility, multiple angles preferred",
            "aspect_ratio": "1:1",
            "size": "1500x1500px (minimum 800x800px)",
            "format": "JPEG",
            "color_space": "sRGB"
        },
        "notes": "Scaling only — do not crop. Add background fill if resizing is required."
    },
    "Zomato": {
        "description": "Zomato food delivery listing",
        "requirements": {
            "composition": "Food must be appetizing, styled, with minimal background",
            "focus": "Highlight freshness, color, and texture",
            "aspect_ratio": "1:1",
            "size": "1080x1080px",
            "format": "JPEG",
            "color_space": "sRGB"
        },
        "notes": "Rescale without cropping dish. Pad or extend background to square if needed."
    },
    "Swiggy": {
        "description": "Swiggy food delivery listing",
        "requirements": {
            "composition": "Dish centered, vibrant colors, professional food presentation",
            "focus": "Boost hunger appeal with strong lighting and contrast",
            "aspect_ratio": "1:1",
            "size": "1080x1080px",
            "format": "JPEG",
            "color_space": "sRGB"
        },
        "notes": "Maintain full dish visibility — rescale, don’t crop. Extend clean background if needed."
    },
    "Instagram Post": {
        "description": "Instagram feed post",
        "requirements": {
            "composition": "High aesthetic value, brand-consistent styling",
            "aspect_ratio": "1:1 (square) or 4:5 (portrait)",
            "size": "1080x1080px (square) / 1080x1350px (portrait)",
            "format": "JPEG",
            "color_space": "sRGB"
        },
        "notes": "Reframe using scaling, not cropping. Fill edges with blurred/extended background if necessary."
    },
    "Instagram Story": {
        "description": "Instagram Story format",
        "requirements": {
            "composition": "Vertical, space reserved for overlay text at top and bottom 20%",
            "aspect_ratio": "9:16",
            "size": "1080x1920px",
            "format": "JPEG or MP4",
            "color_space": "sRGB"
        },
        "notes": "Do not crop subject. Scale to vertical, extend or blur background for fit."
    },
    "OLX": {
        "description": "OLX classified ad listing",
        "requirements": {
            "background": "Neutral (white/gray), clutter-free",
            "composition": "Product shown from multiple angles, honest condition",
            "prohibited": ["personal information", "watermarks", "contact numbers"],
            "aspect_ratio": "1:1 or 4:3",
            "size": "1200x1200px or 1200x900px",
            "format": "JPEG",
            "color_space": "sRGB"
        },
        "notes": "Resize proportionally, no subject cropping. Extend neutral background where needed."
    },
    "Spotify Album Cover": {
        "description": "Spotify album or track artwork",
        "requirements": {
            "composition": "Square image representing album/track identity, visually striking and brand-consistent",
            "aspect_ratio": "1:1",
            "size": "3000x3000px (minimum 640x640px)",
            "format": "JPEG or PNG",
            "color_space": "sRGB"
        },
        "notes": "Must not include URLs, logos, or social media handles. Resize by scaling only, no cropping; extend/pad background if needed."
    }
}

# Perfected Gemini Prompts (With Scaling Rule Included)
PROMPTS = {
    "Amazon": (
        "TASK: Transform this product image to meet Amazon’s main image requirements.\n"
        "MANDATORY CHANGES:\n"
        "- Background: pure white (#FFFFFF / RGB 255,255,255).\n"
        "- Product fills ~85–90% of square frame (1:1).\n"
        "- Remove all props, text, logos, and watermarks.\n"
        "- Apply professional lighting, minimal shadows.\n"
        "- Preserve true product colors and identifiable details.\n"
        "RESIZING RULE:\n"
        "- Resize only by scaling proportionally. Do not crop subject. Extend white background if required to achieve 2000x2000px.\n"
        "OUTPUT: JPEG in sRGB, 2000x2000px. Return ONLY the edited image bytes."
    ),

    "Flipkart": (
        "TASK: Edit this product photo for Flipkart compliance.\n"
        "MANDATORY CHANGES:\n"
        "- White/light-gray background (RGB ~245,245,245).\n"
        "- Product centered, ≥80% frame coverage.\n"
        "- Aspect ratio: 1:1, size 1500x1500px.\n"
        "- Remove watermarks/logos/text.\n"
        "RESIZING RULE:\n"
        "- Resize by scaling, not cropping. Add white/light-gray padding as needed.\n"
        "OUTPUT: JPEG in sRGB. Return ONLY the edited image bytes."
    ),

    "Zomato": (
        "TASK: Enhance this food image for Zomato listing.\n"
        "MANDATORY CHANGES:\n"
        "- Square crop (1:1), 1080x1080px.\n"
        "- Emphasize appetizing look: vibrancy, contrast, texture.\n"
        "- Minimize background distractions.\n"
        "RESIZING RULE:\n"
        "- Scale dish proportionally. Do not crop food. Pad/extend background to fit square.\n"
        "OUTPUT: JPEG in sRGB. Return ONLY the edited image bytes."
    ),

    "Swiggy": (
        "TASK: Edit this food photo for Swiggy.\n"
        "MANDATORY CHANGES:\n"
        "- 1:1 aspect, 1080x1080px.\n"
        "- Enhance vibrancy, lighting, contrast.\n"
        "- Remove clutter around dish.\n"
        "RESIZING RULE:\n"
        "- Keep dish fully visible. Scale only; extend/blur background for fit.\n"
        "OUTPUT: JPEG in sRGB. Return ONLY the edited image bytes."
    ),

    "Instagram Post": (
        "TASK: Adapt this image for Instagram feed.\n"
        "MANDATORY CHANGES:\n"
        "- Square (1080x1080px) or Portrait (4:5 at 1080x1350px).\n"
        "- Apply creative grading, contrast, and clarity.\n"
        "- Subject must remain recognizable.\n"
        "RESIZING RULE:\n"
        "- Reframe without cropping subject. Use padding or blurred background fill if needed.\n"
        "OUTPUT: JPEG in sRGB. Return ONLY the edited image bytes."
    ),

    "Instagram Story": (
        "TASK: Convert this image into Instagram Story format.\n"
        "MANDATORY CHANGES:\n"
        "- Vertical 9:16 at 1080x1920px.\n"
        "- Keep subject centered in safe area (exclude top/bottom 20%).\n"
        "- Add fill/blur background as required.\n"
        "RESIZING RULE:\n"
        "- Scale only, no cropping. Extend background to fit full vertical frame.\n"
        "OUTPUT: JPEG or MP4 in sRGB. Return ONLY the edited media bytes."
    ),

    "OLX": (
        "TASK: Prepare this image for OLX listing.\n"
        "MANDATORY CHANGES:\n"
        "- Neutral white/gray background, clutter removed.\n"
        "- Honest product representation, clear lighting.\n"
        "- Aspect: 1:1 (1200x1200px) or 4:3 (1200x900px).\n"
        "- No personal info, watermarks, or text.\n"
        "RESIZING RULE:\n"
        "- Scale product proportionally. Extend background as required; do not crop.\n"
        "OUTPUT: JPEG in sRGB. Return ONLY the edited image bytes."
    ),

    "Spotify Album Cover": (
        "TASK: Create/edit an album cover image for Spotify.\n"
        "MANDATORY CHANGES:\n"
        "- Square format (1:1) at 3000x3000px (minimum 640x640px).\n"
        "- Strong, artistic design aligned with music identity.\n"
        "- No text overlays, logos, URLs, or social media handles.\n"
        "RESIZING RULE:\n"
        "- Resize by scaling only, do not crop subject. Extend/pad background if needed.\n"
        "OUTPUT: JPEG or PNG in sRGB. Return ONLY the edited image bytes."
    )
}

# Function to encode image
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def safe_rerun():
    """Try to rerun Streamlit reliably across versions.

    Attempts to call st.experimental_rerun(); if not available, toggle
    a session_state timestamp to force UI refresh.
    """
    try:
        st.experimental_rerun()
    except Exception:
        ts_key = "_refresh_ts"
        st.session_state[ts_key] = time.time()


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

    st.markdown("---")
    st.header("Platform Specifications")

# Dropdown to select platform
    selected_platform = st.selectbox("View requirements for:", list(PLATFORMS.keys()))
    platform_info = PLATFORMS[selected_platform]

# Display platform details
    st.write(f"### {selected_platform}")
    st.write(f"**Description:** {platform_info['description']}")

# Requirements (dict pretty print)
    st.subheader("Requirements")
    for key, value in platform_info["requirements"].items():
        st.write(f"- **{key.capitalize()}**: {value}")

# Notes (if available)
    if "notes" in platform_info and platform_info["notes"]:
        st.subheader("Additional Notes")
        st.write(platform_info["notes"])


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

        # Display thumbnails for all uploaded files (we will process all uploads automatically)
        cols = st.columns(len(uploaded_files))
        for i, uf in enumerate(uploaded_files):
            try:
                thumb = Image.open(uf)
                cols[i].image(thumb, use_container_width=True)
                cols[i].write(uf.name)
            except Exception:
                cols[i].write(uf.name)

        # Always process all uploaded images
        images_to_process = uploaded_files

    # (Thumbnails already displayed above) - no separate preview to avoid duplication
        # number of images per platform (moved from sidebar)
        num_images = st.slider("How many images per platform?", 1, 4, 1)

        st.header("Select Platforms")
        selected_platforms = []
        for platform in PLATFORMS.keys():
            if st.checkbox(platform, key=platform):
                selected_platforms.append(platform)
        
        # Optional global extra instructions (displayed above the Generate button)
        st.subheader("Extra Instructions (optional)")
        ei = st.text_area(
            "Enter any extra instructions to apply to all generations (optional)",
            value=st.session_state.get('extra_instructions', ''),
            height=120,
            key="extra_instructions_input",
        )
        st.session_state.extra_instructions = ei

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
                                        platform_size = PLATFORMS[platform].get('size', PLATFORMS[platform].get('requirements', {}).get('size', ''))
                                        w, h = parse_size(platform_size)
                                        generated_list = []
                                        for i in range(num_images):
                                            angle = ANGLES[i]
                                            # Append any platform extras from session state
                                            extra = ""
                                            se = st.session_state.get('spotify_extra', '')
                                            oe = st.session_state.get('olx_extra', '')
                                            # Global optional extra instructions (user-provided)
                                            gen_extra = st.session_state.get('extra_instructions', '')
                                            if platform == 'Spotify Canvas' and se:
                                                extra = f" Include: {se}."
                                            if platform == 'OLX' and oe:
                                                extra = f" Include: {oe}."
                                            if gen_extra:
                                                # Append general extra instructions for all platforms
                                                extra = f" {extra} Additional instructions: {gen_extra}."

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
                        st.image(img, use_container_width=True)
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
                                    safe_rerun()
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
                                            st.image(refreshed, use_container_width=True)
                                        except Exception:
                                            pass
                                        # Try to refresh once more so the UI shows the new image
                                        try:
                                            safe_rerun()
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
                    st.image(img, use_container_width=True)
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
3. **Generate** optimized images using Gemini
4. **Modify (optional)**: choose a generated image and use the "Modify / Chat about this image" box to request an edit — the app will send the image plus your text instruction to Gemini and replace the image in-place when done
5. **Download** your final, formatted assets

### Notes
- The app uses the Gemini image-edit model to perform transformations. It saves generated outputs into a local `generated/` folder.
""")