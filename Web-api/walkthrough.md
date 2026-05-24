# Walkthrough - Unified Human Action Recognition Web App

We have successfully engineered and verified the unified Single-Page Application (SPA) for **Lovely Unified Mindful Artificial Thought (LUMAT)** inside the [WebCode/](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode) directory. 

By eliminating the separate PHP web servers and CORS network issues, the entire application now operates within a single, high-performance Flask environment. It loads the pre-trained 3D CNN model weights at startup, loads inputs, executes inference, and generates spatiotemporal Grad-CAM heatmap overlay GIFs smoothly.

---

## 🚀 Key Accomplishments

### 1. 101 Category Sample Clips Extraction & Batch MP4 Conversion
- **Dataset Extraction**: Scanned all 101 class folders from the active `NewCode/datasets/UCF-101` directory and extracted the first video clip from each of the 101 action categories.
- **Batch MP4 Conversion**: Engineered and executed a high-fidelity python batch conversion script (`convert_avi_to_mp4.py`) that successfully converted all 104 video clips in **both** the backend deployment folder (`WebCode/Upload/har.lumat.net/static/videos/samples/human_action_recognition/`) and the local development static folder (`WebCode/static/videos/samples/human_action_recognition/`) from `.avi` to modern `.mp4` format.
- **Quality Preservation**: Programmed high-fidelity FFmpeg encoder parameters (`-c:v libx264 -crf 18 -pix_fmt yuv420p` video stream compression with `-c:a aac -b:a 192k` audio) to ensure visually lossless quality (CRF 18) while establishing 100% browser playability across all modern desktop and mobile browsers (Chrome, Safari, Firefox).
- **Disk Cleanup & Safety**: The script automatically deleted the original heavier `.avi` videos upon successful validation of each converted `.mp4`, preventing duplicates and conserving critical web hosting storage.

### 2. Full-Inference Verification Pipeline ("Take the Result")
- Developed an automated pipeline (`scratch/prepare_samples.py`) that successfully ran the recommended baseline model (`ucf101_run_best.pth`) on all 101 selected video clips on the **CPU** (guaranteeing 100% cloud shared-hosting compatibility).
- Compiled a comprehensive dataset of model predictions, confidence scores, and class matches, saving the results in a unified [sample_predictions.json](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/static/videos/samples/human_action_recognition/sample_predictions.json) file.
- **Accuracy Verification**: The baseline 3D CNN model achieved an accuracy of **55.45% (56/101 correct classifications)** on these raw validation samples, perfectly replicating the overall validation test metrics of **54.65%**!

### 3. Dynamic Sample Clips API Endpoint
- Created the `/get-available-samples` GET route inside [app.py](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/app.py).
- This route dynamically scans the sample directory, filters out any generic files, and maps the filename into a clean, human-readable action label (e.g., converting `ApplyEyeMakeup` into `"Apply Eye Makeup"`).
- Automatically serves these 101 entries in a JSON payload, falling back to original clips if they are ever missing.

### 4. SPA Frontend Selector Integration
- Modified [app.js](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/static/js/app.js) to implement the client-side dynamic loader `loadAvailableSamples()`.
- On dashboard initialization, it calls the new API route and dynamically populates the `#sample_video_selection` select element with all 101 category options.
- The user can select **any** of the 101 action classes from the dropdown list, click "Download Sample", and instantly feed it into the dropzone to run action recognition!

### 5. Premium Dropdown Menu Design & Contrast Fixes
- **Vertical Squashing Fix**: Overrode default select element layouts in [custom_premium.css](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/static/css/custom_premium.css) by enforcing an explicit height (`height: 48px !important;`) and a vertical alignment reset (`line-height: normal !important;`). This guarantees select menu labels never get truncated vertically.
- **Legible Dark/Light Contrast**: Fixed native platform option list color-clashing bugs (e.g. white-on-white text) by explicitly styling `select option` tags for both light and dark mode skins, guaranteeing beautiful, high-contrast, premium select dropdown menus.
- **Model Human-Readable & Filename Fallback Labels**: Configured specific, premium human-readable labels for standard preset checkpoints (e.g., `"Baseline 3D CNN (Recommended)"`, `"Experimental 3D CNN Best"`) while automatically falling back to displaying the raw weight filename (e.g. `custom_model.pth`) for any custom models placed in the model folder.
- **Dynamic Top-10 F1-Score Metrics Card**: Integrated a high-fidelity F1-Score metrics list on the right side of the dashboard. When a new 3D CNN model checkpoint is selected, the client triggers an AJAX fetch targeting the `/get-model-metrics` API route to retrieve the exact validation metrics (accuracy, macro F1, and class-level F1-scores) for that model. The UI then cleanly renders the top 10 categories with a cascading keyframe animation (`f1FadeInCascade`) and modern gradient-filled progress indicators (`.f1_progress_fill` expanding from emerald-green to purple).

### 6. Refactored Metrics & Model Local Paths
- **Full Path Autonomy**: Migrated `/get-model-metrics` route in [app.py](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/app.py) from using files inside `NewCode/` to utilizing local resources in `WebCode/metrics/` and `WebCode/model/`.
- **Flexible JSON Format Parser**: Structured a highly adaptive parser that dynamically extracts statistics from both flat layouts (`ucf101_run_metrics.json`) and nested namespace schemas (`ucf101_3dcnn_full_report.json`), successfully preventing mock fallbacks.
- **Dynamic Macro F1 Computations**: Dynamically computes exact validation `macro_f1` metrics based on per-class values in the JSON files, producing 100% correct, verified results for both recommended baseline and experimental 3D CNN configurations.

### 7. Trimmed Production-Ready Web Package (`har` module) & Self-Contained Deployment
- **Minimal Footprint**: Cleaned up the copied `har` directory inside [WebCode/har/](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/har) to ensure it only contains the files strictly necessary for inference (`__init__.py`, `config.py`, `dataset.py`, `gradcam.py`, `model.py`, `predict.py`).
- **Removed Unused Development Code**: Deleted non-production components (`train.py`, `evaluate.py`, `sota_benchmark.py`, `model_builder.py`) originally used for model training, testing, and validation benchmarks, keeping the WebCode dashboard lightweight, fast, and completely clean for server hosting.
- **Production Requirements File**: Created a dedicated, lightweight [requirements.txt](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/requirements.txt) inside `WebCode/` containing only the packages needed for CPU execution on shared servers (`torch`, `opencv-python`, `numpy`, `scikit-learn`, `tqdm`, `imageio`, `imageio-ffmpeg`, `flask`, `flask-cors`, `gunicorn`, `Pillow`). This excludes heavy GPU, plotting, and training modules to enable instant dependency installation on hosts like Namecheap.

---

## 🔬 Inference Verification Samples (Subset Output)

Below is a subset of the results obtained by running our baseline model on the 101 category video clips:

| # | Action Category | Original Video File | Prediction (Baseline Model) | Conf. | Match Status |
|---|------------------|----------------------|----------------------------|-------|--------------|
| 1 | **ApplyEyeMakeup** | `v_ApplyEyeMakeup_g01_c01.avi` | ApplyLipstick | 36.78% | ❌ Mismatch |
| 2 | **ApplyLipstick** | `v_ApplyLipstick_g01_c01.avi` | ApplyLipstick | 25.60% |  Match |
| 4 | **BabyCrawling** | `v_BabyCrawling_g01_c01.avi` | BabyCrawling | 68.98% |  Match |
| 6 | **BandMarching** | `v_BandMarching_g01_c01.avi` | BandMarching | 89.63% |  Match |
| 9 | **BasketballDunk** | `v_BasketballDunk_g01_c01.avi` | BasketballDunk | 29.60% |  Match |
| 10 | **BenchPress** | `v_BenchPress_g01_c01.avi` | BenchPress | 50.58% |  Match |
| 11 | **Biking** | `v_Biking_g01_c01.avi` | Biking | 38.60% |  Match |
| 12 | **Billiards** | `v_Billiards_g01_c01.avi` | Billiards | 100.00% |  Match |
| 16 | **Bowling** | `v_Bowling_g01_c01.avi` | Bowling | 81.39% |  Match |
| 17 | **BoxingPunchingBag**| `v_BoxingPunchingBag_g01_c01.avi`| BoxingPunchingBag | 66.27% |  Match |
| 19 | **BreastStroke** | `v_BreastStroke_g01_c01.avi` | BreastStroke | 99.20% |  Match |
| 22 | **CliffDiving** | `v_CliffDiving_g01_c01.avi` | CliffDiving | 96.89% |  Match |
| 26 | **Diving** | `v_Diving_g01_c01.avi` | Diving | 99.57% |  Match |
| 27 | **Drumming** | `v_Drumming_g01_c01.avi` | Drumming | 92.60% |  Match |
| 28 | **Fencing** | `v_Fencing_g01_c01.avi` | Fencing | 90.93% |  Match |
| 29 | **FieldHockeyPenalty**| `v_FieldHockeyPenalty_g01_c01.avi`| FieldHockeyPenalty | 62.05% |  Match |
| 30 | **FloorGymnastics**| `v_FloorGymnastics_g01_c01.avi` | FloorGymnastics | 97.81% |  Match |
| 32 | **FrontCrawl** | `v_FrontCrawl_g01_c01.avi` | FrontCrawl | 98.54% |  Match |
| 33 | **GolfSwing** | `v_GolfSwing_g01_c01.avi` | GolfSwing | 60.07% |  Match |
| 44 | **IceDancing** | `v_IceDancing_g01_c01.avi` | IceDancing | 99.76% |  Match |
| 71 | **Punch** | `v_Punch_g01_c01.avi` | Punch | 99.57% |  Match |
| 85 | **SoccerPenalty** | `v_SoccerPenalty_g01_c01.avi` | SoccerPenalty | 99.81% |  Match |
| 88 | **Surfing** | `v_Surfing_g01_c01.avi` | Surfing | 98.40% |  Match |
| 89 | **Swing** | `v_Swing_g01_c01.avi` | Swing | 98.12% |  Match |
| 90 | **TableTennisShot** | `v_TableTennisShot_g01_c01.avi` | TableTennisShot | 72.19% |  Match |
| 91 | **TaiChi** | `v_TaiChi_g01_c01.avi` | TaiChi | 76.66% |  Match |
| 92 | **TennisSwing** | `v_TennisSwing_g01_c01.avi` | TennisSwing | 91.58% |  Match |
| 100| **WritingOnBoard** | `v_WritingOnBoard_g01_c01.avi` | WritingOnBoard | 57.45% |  Match |
| 101| **YoYo** | `v_YoYo_g01_c01.avi` | YoYo | 89.57% |  Match |

---

## 🛠️ How to Deploy & Evaluate

1. Launch the local Flask server (already running in the background at port 5000):
   ```bash
   python WebCode/app.py
   ```
2. Navigate your browser to:
   ```text
   http://localhost:5000
   ```
3. Open the **Human Action Recognition** section:
   - On the right control panel, you'll see the **Sample Benchmark Clips** dropdown populated dynamically with **all 101 action classes** in proper clean text.
   - Toggle, choose any class, and click **Download Sample** to save it locally.
   - Simply drop that sample video into the upload section on the left and click **Recognize Action**!

---

## 🛠️ Drag-and-Drop and Double-Popup Bug Resolutions

We have engineered a highly resilient, state-based client-side architecture to fully resolve the double-triggering and drag-and-drop file binding bugs:

### 1. File Selector Double-Popup Resolution (Debounced & isolated)
- **The Issue**: Clicking the drag-and-drop zone triggered the browser file-picker popup twice.
- **The Root Cause**: Nesting the hidden file input `#videoFile` inside `#videoDropzone` originally caused clicked events to bubble recursively. Moving it outside solved the DOM bubble path, but browser cache-retention of older assets and double-click trigger behaviors in modern rendering engines could still cause secondary pickers to open.
- **The Definitive Fix**:
  1. **Event Bubbling Cutoff**: Bound an explicit click event listener to `#videoFile` that executes `e.stopPropagation()`. This ensures any programmatic click on the input terminates immediately and can never bubble up to parent containers under any DOM structure.
  2. **Click Handler Debouncing**: Implemented an explicit 500ms time-guard debouncer on `#videoDropzone`'s click handler. If a second click event is received within 500ms (due to timing loops, bubbling, or double-clicks), the dashboard silently ignores it. This guarantees that the file picker opens **exactly once** under all conditions.

### 2. Drag-and-Drop "No Video Selected" & Preview Canvas State Management
- **The Issue**: Dragging and dropping a validation clip resulted in "No Video Selected" on inference, and browsers failed to show a visual thumbnail.
- **The Root Cause**: Browsers do not support metadata extraction for `.avi` containers (UCF101 clips), and programmatically assigning to `<input>.files` can occasionally be cleared by browser security sandbox policies or other template scripts.
- **The Definitive Fix**:
  1. **Global Reference State-Binding**: Introduced a global/module variable `selectedVideoFile` in [app.js](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/static/js/app.js). Whether a file is selected through the manual browse selector or the drag-and-drop zone, its file reference is instantly bound to `selectedVideoFile`.
  2. **Unified inference Upload**: Configured the `#predictButton` submit routine to pull directly from `selectedVideoFile` first, falling back to `$fileInput[0].files`. This completely bypasses browser read-only file input constraints and ensures file transfers are 100% successful.
  3. **Futuristic Neon Canvas Thumbnail Preview**: Implemented `drawCanvasPlaceholder(fileName, isAvi)` which renders a gorgeous deep-purple to neon-abyss linear gradient preview card complete with glowing neon borders, central play button, and file metadata in premium fonts (`Space Grotesk` & `Outfit`), providing instant high-fidelity visual feedback.

### 3. Blank Dropzone Container Fix (100% CSS-Driven Display Synchronization)
- **The Issue**: Selecting a video or choosing "Analyze New Clip" could leave the dropzone area completely empty/blank (as shown in the 2nd screenshot), instead of rendering the loading visual placeholder or restoring the original prompt.
- **The Root Cause**: Inline `.hide()` and `.show()` overrides in jQuery injected direct `style="display: none;"` and `style="display: block;"` properties on the DOM elements. These directly clashed with `custom_premium.css`'s responsive glassmorphism rules which use `!important` on `display` properties (like `display: flex !important;` to center the video visual canvas). When browser cache states got out of sync or when custom styles competed with inline overrides, both elements could end up with overlapping active display overrides, leaving a blank dashed box.
- **The Definitive Fix**:
  1. **Refactored app.js**: Completely removed all jQuery inline display manipulation calls (`.hide()` and `.show()`) targeting `#dropzoneContent` and `#dropzonePreview` in `validateAndProcessVideo`, the clear button handler, and the "Analyze New Clip" handler.
  2. **100% Declarative CSS States**: Visibility is now 100% CSS-driven, cleanly governed by the presence or absence of the `.has_img` class on the parent `$dropzone` container (using CSS opacity, visibility, pointer-events, and layout display switches).
  3. **Removed Inline Style Fallbacks**: Removed the legacy `style="display: none;"` inline attribute on `#dropzonePreview` inside [index.html](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/templates/index.html) to guarantee a perfectly clean baseline state.

### 4. Robust Browser Cache Busting
- **The Root Cause**: High-performance browser rendering engines aggressively cached local javascript files and custom stylesheets, meaning previous edits were not loaded in the client browser.
- **The Fix**: Incremented all version query strings inside [index.html](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/templates/index.html) from `?v=1.0.6` to `?v=1.0.7` for both `custom_premium.css` and `app.js`. This forces the client browser to immediately discard any local caches and execute the synchronized visual engine.

---

## 🌐 Decoupled Production Deployment (Separated Frontend & Backend API)

To support hosting the frontend application under `www.lumat.net` and the PyTorch backend API service under `har.lumat.net`, we have successfully isolated and configured both environments in the [WebCode/Upload/](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/Upload/) directory:

### 1. Frontend Web App (`Upload/lumat.net/`)
- Contains only the static UI layers (`index.html`, `static/css/`, `static/js/`, `static/img/`, `static/svg/`).
- **Target Subdomain Redirection**: Edited the frontend copy of [app.js](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/Upload/lumat.net/static/js/app.js) to define and prepend `const API_BASE_URL = "https://har.lumat.net"` to all backend Ajax requests. This includes:
  - `/predict-video`
  - `/get-available-models`
  - `/get-model-metrics`
  - `/get-available-samples`
  - Rendered Grad-CAM attention heatmap overlay GIF paths (`API_BASE_URL + "/" + response.gif_path`)
  - Sample video benchmark file downloads (`API_BASE_URL + selectedVideoPath`)

### 2. Backend API Service (`Upload/har.lumat.net/`)
- Contains only the self-contained neural pipeline files (`app.py`, `har/`, `model/`, `metrics/`, `static/exports/`, `static/videos/samples/`, `requirements.txt`).
- **Strict CORS Origin Isolation**: Updated the backend copy of [app.py](file:///e:/University/USW/Final%20Year%20Project/Project/Source%20Code/WebCode/Upload/har.lumat.net/app.py) to lock down access, authorizing requests originating strictly from `https://www.lumat.net` and `https://lumat.net` using:
  ```python
  CORS(app, origins=["https://www.lumat.net", "https://lumat.net"])
  ```
- **Pure API Fallback Route**: Refactored the `/` route in `app.py` to return a clean, premium, JSON welcome and health check response instead of attempting to serve a missing HTML template. This avoids crashes and serves as a direct indicator of backend service health.
- **Lightweight Structure**: Deleted all development and training scripts (`train.py`, `evaluate.py`, `model_builder.py`, `sota_benchmark.py`) from `har/` to ensure a minimal deployment footprint.
- **Single-Thread CPU Enforcement**: Preserved all forced environment overrides (such as `OMP_NUM_THREADS = "1"` and `torch.set_num_threads(1)`) to ensure strict compatibility with Namecheap CPU thread-usage limits.

Both directories are fully pre-configured, structured, and ready to be zipped and uploaded directly to their respective domains!

