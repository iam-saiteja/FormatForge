FormatForge — README

Overview

FormatForge is a Streamlit application that uses the Gemini image editing model to convert a single uploaded image into platform-ready assets (Amazon, Flipkart, Zomato, Swiggy, Instagram feed/story, OLX, Spotify album cover). The app accepts up to four source images, generates 1–4 variations per selected platform, and allows in-place modification of any generated image via a simple "Modify / Chat about this image" text box.

This README documents how the app works, how to run it locally, the code structure and important implementation details, and how the modify-workflow operates.

Contents

- Quick start
- UI walkthrough
- Generation flow and prompts
- Modify / re-edit workflow
- Files and important functions
- Platform specifications
- Security and limitations
- Troubleshooting

Quick start

1. Clone the repository (first step):

```powershell
git clone https://github.com/iam-saiteja/SKU-Ready
cd "SKU-Ready"
```

2. Ensure Python 3.10+ is installed and create/activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

3. Install required packages using the provided `requirements.txt`:

```powershell
pip install -r requirements.txt
```

4. Run the app:

```powershell
streamlit run work.py
```

5. Open the URL printed by Streamlit (usually http://localhost:8501) and enter your Gemini API key in the sidebar.

UI walkthrough

- Sidebar: enter your Gemini API key and inspect per-platform specifications.
- Left column: upload up to 4 images (jpg/png), pick platforms, choose how many images per platform (1–4), and optionally enter "Extra Instructions" (applies to all generations). Click *Generate Formatted Images* to start.
- Right column: view generated assets grouped by platform. For each generated image you can:
  - Download the image
  - Enter a modification instruction in the "Modify / Chat about this image" box and click "Apply modification" to re-edit that generated image in-place.

Generation flow and prompts

- The app encodes uploaded images to base64 and calls the Gemini image-editing model using the `client.models.generate_content` pattern with the model `gemini-2.5-flash-image-preview`.
- Each platform has a strict prompt in `PROMPTS` containing mandatory changes and a resizing rule ("scale only, do not crop the subject"). When generating multiple angles, the app appends a simple angle hint such as "front view" or "left 45 degree angle".
- If the optional Extra Instructions field is filled, its text is appended to the prompt for every generation.

Modify / re-edit workflow

- Each generated image has a "Modify / Chat about this image" input. When you enter instructions and click *Apply modification*:
  - The app marks the item busy and queues the modification.
  - On rerun, the queued instruction is sent to the Gemini model along with the exact generated image as the input image.
  - When the model returns an edited image, the app overwrites the same file on disk and updates the session-state entry for that image (flagged `modified=True`).
  - The UI displays the updated image immediately.

Files and important functions

- `work.py` — main Streamlit app. Key functions:
  - `encode_image(image: PIL.Image) -> str` — encodes a PIL image to base64.
  - `validate_and_fix_b64(b64_str) -> Optional[str]` — heuristically validates and repairs base64 image strings returned from the model.
  - `call_gemini_api(image_b64, prompt, platform) -> Optional[str]` — calls the Gemini model and returns a base64-encoded image string.
  - `resize_image_file(path, width, height)` — resizes saved files to exact pixel dimensions using Pillow LANCZOS resampling.
  - `safe_rerun()` — attempts to call `st.experimental_rerun()` and falls back to toggling a session timestamp to force re-render.

- `generated/` — directory created by the app at runtime where generated images are saved.

Platform specifications

The `PLATFORMS` dict in `work.py` defines precise requirements per platform, including size, aspect-ratio, and required transformations. The app uses these to both compose prompts and to resize saved images after generation.

Security and limitations

- API keys: enter your Gemini API key in the sidebar. The app stores it in `st.session_state` only for the current browser session and does not persist it to disk.
- Model responses: the app attempts to robustly parse different possible SDK response shapes. However, returned images depend on the model's behavior and prompt tuning.
- Local storage: generated images are saved into a local `generated/` directory in the app folder. Remove or secure this directory as needed.

Troubleshooting

- If Streamlit raises deprecation errors about `use_column_width`, the app uses `use_container_width` instead.
- If rerun behavior does not immediately show updated images, try refreshing the browser; the app attempts to force a rerender but browser caching can interfere.
- If the model returns non-image payloads, check logs printed in the Streamlit UI for debugging messages.

Development notes

- The current implementation focuses on a simple, single-node developer flow. For production: add authentication, server-side storage, rate limits, robust retry/backoff, and user quotas.

Contact

- Email: iamsaitejathanniru@gmail.com
- Website: https://thannirusaiteja.me
- LinkedIn: https://linkedin.com/in/thannirusaiteja

Collaborate

Anyone is welcome to collaborate on this project. Recommended workflow:

1. Fork the repository (`https://github.com/iam-saiteja/SKU-Ready`).
2. Create a descriptive feature branch and make your changes.
3. Open a Pull Request describing the change and any testing steps.
4. Open an Issue first if you prefer to discuss the idea before coding.

For quick collaboration requests or questions, use the email above or open an issue on the repository.

License

This project is provided free of charge under the MIT License. Full credit should be given to the original author "iam-saiteja" when reusing or redistributing the project. See the `LICENSE` file for the full license text.
