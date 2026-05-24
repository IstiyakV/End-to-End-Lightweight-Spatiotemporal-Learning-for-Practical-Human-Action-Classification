/**
 * LUMAT - Single Page Application Interactive JS Engine
 * Core logic for file uploads, drag-and-drop, canvas thumbnail rendering,
 * unified Flask API prediction AJAX calls, and premium toast notifications.
 */

$(document).ready(function () {
    // API base URL configuration (empty for local relative, overridden during build)
    const API_BASE_URL = "";

    // -------------------------------------------------------------
    // 1. Unified Single-Page Application (SPA) Tab Switching Router
    // -------------------------------------------------------------
    function switchTab(tabId) {
        if (!tabId) return;

        console.log(`[Router] Switching view to tab: ${tabId}`);

        // Get target tab element
        const $targetTab = $("#" + tabId);
        if ($targetTab.length === 0) return;

        // Hide all tabs with fade-out
        $(".tab-content").removeClass("active-tab").hide();

        // Show target tab and trigger fade-in transform
        $targetTab.show();
        setTimeout(function () {
            $targetTab.addClass("active-tab");
        }, 30);

        // Update active navigation item styling in sidebar
        $(".leftpanel_content ul.group__list li").removeClass("active");

        if (tabId === "home-tab") {
            $(".menu-item-home").addClass("active");
        } else if (tabId === "har-tab") {
            $(".menu-item-har").addClass("active");
        }

        // Close mobile panels or dropdown menus if open
        $("body").removeClass("panel-opened");
        
        // Scroll page up to coordinate smooth transition
        $("html, body").animate({ scrollTop: 0 }, "slow");
    }

    // Bind tab triggers (clickable elements with fn__tab_trigger or data-tab/data-target bindings)
    $(document).on("click", ".fn__tab_trigger", function (e) {
        e.preventDefault();
        const target = $(this).attr("data-target") || $(this).attr("data-tab");
        switchTab(target);
    });

    // Search item selection triggers tab switch
    $(document).on("click", ".search-item", function (e) {
        e.preventDefault();
        const tab = $(this).data("tab");
        switchTab(tab);
        $(".techwave_fn_searchbar").removeClass("active"); // close search bar
    });


    // -------------------------------------------------------------
    // 2. Custom Premium Notification System (Toast Modals)
    // -------------------------------------------------------------
    function showToast(title, desc, type = "info") {
        const toastId = "toast_" + Date.now();
        let iconSvg = "/static/svg/info.svg";
        
        if (type === "success") {
            iconSvg = "/static/svg/check.svg";
        } else if (type === "error") {
            iconSvg = "/static/svg/close.svg";
        }

        const toastHtml = `
            <div class="premium_toast ${type}" id="${toastId}">
                <div class="toast_icon">
                    <img src="${iconSvg}" class="fn__svg" style="filter: brightness(0) invert(1);" alt="${type}">
                </div>
                <div class="toast_content">
                    <h4 class="toast_title">${title}</h4>
                    <p class="toast_desc">${desc}</p>
                </div>
            </div>
        `;

        $("#toastWrapper").append(toastHtml);

        // Parse svg inline if frenify init function exists
        if (window.FrenifyTechWave && typeof window.FrenifyTechWave.img_to_svg === "function") {
            window.FrenifyTechWave.img_to_svg();
        }

        // Trigger slide-in
        setTimeout(function () {
            $("#" + toastId).addClass("show");
        }, 50);

        // Auto remove toast after 4.5 seconds
        setTimeout(function () {
            $("#" + toastId).removeClass("show");
            setTimeout(function () {
                $("#" + toastId).remove();
            }, 400);
        }, 4500);
    }

    // Wire Coming Soon notifications
    $(document).on("click", ".coming-soon-trigger", function (e) {
        e.preventDefault();
        e.stopPropagation();
        showToast(
            "Feature Upcoming",
            "This module is undergoing optimization and will be activated in the next development cycle.",
            "info"
        );
    });

    // Wire Log Out trigger
    $(document).on("click", ".logout-trigger", function (e) {
        e.preventDefault();
        showToast(
            "Session Active",
            "LUMAT is running in demo evaluation mode. No active session to clear.",
            "success"
        );
    });


    // -------------------------------------------------------------
    // 3. Drag and Drop + Local File Selection
    // -------------------------------------------------------------
    const $dropzone = $("#videoDropzone");
    const $fileInput = $("#videoFile");

    // Globally hold the currently active file object (bypasses browser input read-only restrictions)
    let selectedVideoFile = null;
    let lastDropzoneClickTime = 0;

    // Click dropzone triggers file dialog
    $dropzone.on("click", function (e) {
        e.preventDefault();
        e.stopPropagation();

        const now = Date.now();
        if (now - lastDropzoneClickTime < 500) {
            console.log("[Dropzone] Double-trigger ignored via click debouncer.");
            return;
        }
        lastDropzoneClickTime = now;

        if (!$(e.target).closest("#clearVideoFile").length && !$(e.target).closest(".canvas_wrapper").length) {
            console.log("[Dropzone] Programmatically opening native file selector...");
            $fileInput[0].click();
        }
    });

    // Prevent hidden input click event from bubbling up to parent dropzone container (structural double-trigger prevention)
    $fileInput.on("click", function (e) {
        e.stopPropagation();
    });

    // Drag-over styling shifts
    $dropzone.on("dragover dragenter", function (e) {
        e.preventDefault();
        e.stopPropagation();
        $dropzone.addClass("dragover");
    });

    $dropzone.on("dragleave", function (e) {
        e.preventDefault();
        e.stopPropagation();
        $dropzone.removeClass("dragover");
    });

    // Handle video file drop
    $dropzone.on("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        $dropzone.removeClass("dragover");
        const dt = e.dataTransfer || (e.originalEvent && e.originalEvent.dataTransfer);
        const files = dt ? dt.files : [];
        if (files.length > 0) {
            validateAndProcessVideo(files[0]);
        }
    });

    // Handle normal browse change
    $fileInput.on("change", function () {
        if (this.files.length > 0) {
            validateAndProcessVideo(this.files[0]);
        }
    });

    // Draw premium visual placeholder for videos (especially for formats like AVI that modern browsers cannot natively play)
    function drawCanvasPlaceholder(fileName, isAvi = false) {
        const canvas = document.getElementById("videoThumbnailCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        
        // Premium card size
        canvas.width = 400;
        canvas.height = 250;

        // Elegant gradient dark background
        const grad = ctx.createLinearGradient(0, 0, 400, 250);
        grad.addColorStop(0, '#1a103c'); // Modern deep purple
        grad.addColorStop(1, '#0b071a'); // Dark neon abyss
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 400, 250);

        // Neon glowing outer border
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, 396, 246);

        // Circular background for play icon
        ctx.fillStyle = 'rgba(168, 85, 247, 0.15)';
        ctx.beginPath();
        ctx.arc(200, 100, 45, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
        ctx.lineWidth = 3;
        ctx.stroke();

        // White play icon triangle
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.moveTo(190, 80);
        ctx.lineTo(220, 100);
        ctx.lineTo(190, 120);
        ctx.closePath();
        ctx.fill();

        // Premium text details
        ctx.font = "bold 15px 'Space Grotesk', sans-serif";
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.fillText(isAvi ? "AVI Validation Video Loaded" : "Video Format Loaded", 200, 175);

        // File name text
        ctx.font = "12px 'Outfit', sans-serif";
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        let displayFileName = fileName;
        if (displayFileName.length > 35) {
            displayFileName = displayFileName.substring(0, 15) + "..." + displayFileName.substring(displayFileName.length - 15);
        }
        ctx.fillText(displayFileName, 200, 205);
    }

    // Validate video file formats and generate canvas thumbnail
    function validateAndProcessVideo(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (ext !== 'mp4' && ext !== 'avi') {
            showToast("Invalid Format", "Action Recognition supports .mp4 or .avi formats only.", "error");
            return;
        }

        // Enforce strict 15MB size limit to prevent server overload
        const maxFileSize = 15 * 1024 * 1024; // 15MB
        if (file.size > maxFileSize) {
            showToast("File Too Large", "Please select a video clip smaller than 15 MB.", "error");
            return;
        }

        // Bind selected file to memory reference
        selectedVideoFile = file;

        // Sync file input files collection as backup
        try {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            $fileInput[0].files = dataTransfer.files;
        } catch (e) {
            console.warn("Failed to set dataTransfer files:", e);
        }

        // Show loading progress info
        showToast("Processing Video", "Extracting first sequence frame for visual dashboard preview...", "info");

        $("#selectedFileName").text(file.name);
        $dropzone.addClass("has_img");
        $("#predictButton").fadeIn();

        // Instantly render gorgeous, high-contrast placeholder card
        drawCanvasPlaceholder(file.name, ext === 'avi');

        // Render Canvas Thumbnail from Video Metadata (Only if natively supported, e.g. MP4)
        if (ext === 'mp4') {
            const video = document.createElement("video");
            video.preload = "metadata";
            video.src = URL.createObjectURL(file);
            
            video.onloadedmetadata = function () {
                // Seek to 0.5s or beginning to grab a frame
                video.currentTime = Math.min(0.5, video.duration / 2);
            };

            video.onseeked = function () {
                const canvas = document.getElementById("videoThumbnailCanvas");
                const ctx = canvas.getContext("2d");

                // Set canvas size matching the video resolution aspect ratios
                canvas.width = video.videoWidth || 320;
                canvas.height = video.videoHeight || 240;

                // Draw image on canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Clean up temporary blob resource
                window.URL.revokeObjectURL(video.src);
            };
        }
    }

    // Clear video selection and reset dropzone back to normal
    $(document).on("click", "#clearVideoFile", function (e) {
        e.preventDefault();
        e.stopPropagation();
        
        selectedVideoFile = null;
        $fileInput.val("");
        $("#selectedFileName").text("");
        $dropzone.removeClass("has_img");
        $("#predictButton").fadeOut();
        $("#progressBar").hide();
        $("#progress").css("width", "0%");
        $("#progressPercent").text("0%");
        
        showToast("Video Cleared", "Local selection has been reset.", "info");
    });


    // -------------------------------------------------------------
    // 4. Action Recognition Inference API Integration (AJAX Upload)
    // -------------------------------------------------------------
    // Dynamic Model / Engine layout binder
    $(document).on("change", "#inferenceEngineSelect", function () {
        const engine = $(this).val();
        if (engine === "client") {
            $("#modelSelectionContainer").slideUp(350);
            showToast("Engine Switched", "Using zero-server-cost local in-browser GPU execution (Paper 303K Model).", "success");
        } else {
            $("#modelSelectionContainer").slideDown(350);
            showToast("Engine Switched", "Using cloud server API (supports dynamic checkpoints and Grad-CAM).", "info");
        }
    });

    $("#predictButton").on("click", function (e) {
        e.preventDefault();

        // Retrieve file from global variable reference first, fall back to file input files collection
        const videoFile = selectedVideoFile || ($fileInput[0].files && $fileInput[0].files[0]);
        if (!videoFile) {
            showToast("No Video Selected", "Please drag or browse a video clip to run inference.", "error");
            return;
        }

        const engine = $("#inferenceEngineSelect").val() || "server";
        if (engine === "client") {
            runClientSideONNXInference(videoFile);
            return;
        }

        const formData = new FormData();
        formData.append("file", videoFile);

        // Dynamic model selection parameter mapping
        const selectedModel = $("#neuralModelSelect").val();
        if (selectedModel) {
            formData.append("model_name", selectedModel);
        }

        // UI transitions: shift settings panel out and results display in
        $("#select_container").addClass("hidden-container");
        $("#result_container").show();

        // Reset result displays to loading states
        $("#gifImage").hide().attr("src", "");
        $("#spinnerWrapper").show();
        $("#processingText").show();
        $("#inferenceSuccessDisplay").hide();

        $("#progressBar").show();
        $("#progress").css("width", "0%");
        $("#progressPercent").text("0%");

        showToast("Model Initialized", "Uploading file to local network. Preparing 3D CNN pipeline...", "info");

        // Run AJAX Multi-part form upload targeting Flask backend
        $.ajax({
            url: API_BASE_URL + "/predict-video",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            xhr: function () {
                const xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener("progress", function (e) {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        $("#progress").css("width", percent + "%");
                        $("#progressPercent").text(percent + "%");
                    }
                });
                return xhr;
            },
            success: function (response) {
                // Success: Hide upload progress indicators
                $("#progressBar").hide();
                $("#processingText").hide();
                $("#spinnerWrapper").hide();

                // Format predicted response payload details
                const predictedClass = response.result.predicted_class;
                const confidence = response.result.confidence;
                const confidencePercentage = Math.round(confidence * 100);

                // Show target Grad-CAM attention heatmap overlay
                $("#gifImage").attr("src", API_BASE_URL + "/" + response.gif_path).fadeIn(600);
                
                // Show result text details with fancy glow
                $("#result").html(`Predicted Class: <span style="color: #a855f7;">${predictedClass}</span>`);
                $("#confidenceValueText").text(confidencePercentage + "%");
                
                // Show successful results wrapper
                $("#inferenceSuccessDisplay").fadeIn(400);

                // Animate confidence meter fill
                setTimeout(function () {
                    $("#confidenceBarFill").css("width", confidencePercentage + "%");
                }, 200);

                showToast(
                    "Classification Complete", 
                    `Class: ${predictedClass} with ${confidencePercentage}% accuracy confidence.`, 
                    "success"
                );
            },
            error: function (xhr, status, error) {
                // Error: Reset panel display states
                $("#progressBar").hide();
                $("#processingText").hide();
                $("#spinnerWrapper").hide();
                
                // Display error fallbacks in card UI
                $("#gifImage").attr("src", "/static/img/work.gif").fadeIn();
                $("#result").html(`<span style="color:#ef4444;">Neural Classification Failure</span>`);
                $("#confidenceValueText").text("0%");
                $("#confidenceBarFill").css("width", "0%");
                $("#inferenceSuccessDisplay").fadeIn(400);

                let errorMsg = "Verify PyTorch back-end execution status.";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                
                showToast("Inference Error", errorMsg, "error");
            }
        });
    });

    // Try again buttons reload HAR UI
    $(document).on("click", ".restart_dashboard_btn", function (e) {
        e.preventDefault();
        
        // Reset local variables
        selectedVideoFile = null;
        $fileInput.val("");
        $("#selectedFileName").text("");
        $dropzone.removeClass("has_img");
        $("#predictButton").hide();
        
        // Display toggling
        $("#result_container").hide();
        $("#select_container").removeClass("hidden-container").hide().fadeIn(450);

        showToast("Dashboard Reset", "Prepared for next video inference.", "info");
    });


    // -------------------------------------------------------------
    // 5. Benchmark Sample Videos Download Actions
    // -------------------------------------------------------------
    $("#downloadSample").on("click", function (e) {
        e.preventDefault();

        const selectedVideoPath = $("#sample_video_selection").val();
        if (!selectedVideoPath) return;

        showToast("Downloading", "Transferring validation benchmark sample to downloads directory...", "info");

        // Extract filename from the path
        const filename = selectedVideoPath.split("/").pop();
        
        // Point to the dedicated download-sample endpoint to force attachment download
        const downloadUrl = API_BASE_URL + `/download-sample?filename=${encodeURIComponent(filename)}`;

        const anchor = document.createElement("a");
        anchor.href = downloadUrl;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
    });

    // -------------------------------------------------------------
    // 6. Dynamic Model Checkpoint Selector API Loader
    // -------------------------------------------------------------
    function loadAvailableModels() {
        $.ajax({
            url: API_BASE_URL + "/get-available-models",
            type: "GET",
            dataType: "json",
            success: function (response) {
                if (response && response.models && response.models.length > 0) {
                    const $select = $("#neuralModelSelect");
                    $select.empty();
                    response.models.forEach(function (model) {
                        $select.append(`<option value="${model.filename}">${model.label}</option>`);
                    });
                    console.log(`[Models] Successfully loaded ${response.models.length} dynamic models.`);
                    
                    // Load metrics for the active selected model
                    const activeModel = $select.val();
                    loadModelMetrics(activeModel);
                } else {
                    console.warn("[Models] No models returned from the backend.");
                }
            },
            error: function (xhr, status, error) {
                console.error("[Models] Failed to fetch available models:", error);
                // Fallback option in case of endpoint timeout or error
                const $select = $("#neuralModelSelect");
                $select.empty().append('<option value="ucf101_run_best.pth">Baseline 3D CNN (Recommended)</option>');
                loadModelMetrics("ucf101_run_best.pth");
            }
        });
    }

    // -------------------------------------------------------------
    // 6.5 Dynamic Model Metrics Loader (F1-Scores)
    // -------------------------------------------------------------
    function loadModelMetrics(modelName) {
        if (!modelName) return;
        
        console.log(`[Metrics] Fetching top F1 scores for model: ${modelName}`);
        const $container = $("#f1ScoresContainer");
        
        // Show subtle loading state inside container
        $container.html(`
            <div style="text-align: center; padding: 15px; font-family: 'Space Grotesk', sans-serif; font-size: 13px; color: rgba(255,255,255,0.4);">
                <img src="/static/img/loading_small.gif" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 8px;">
                <span>Loading metrics...</span>
            </div>
        `);
        
        $.ajax({
            url: API_BASE_URL + "/get-model-metrics",
            type: "GET",
            data: { model_name: modelName },
            dataType: "json",
            success: function (response) {
                if (response && response.top_10 && response.top_10.length > 0) {
                    $container.empty();
                    
                    response.top_10.forEach(function (item, index) {
                        const score = item.f1;
                        const scorePercent = Math.round(score * 100);
                        const delay = index * 40; // cascade delay animation
                        
                        const itemHtml = `
                            <div class="f1_item" style="animation-delay: ${delay}ms;">
                                <div class="f1_item_labels">
                                    <span class="f1_class_name">${item.label}</span>
                                    <span class="f1_value">${score.toFixed(2)}</span>
                                </div>
                                <div class="f1_progress_bg">
                                    <div class="f1_progress_fill" data-width="${scorePercent}%" style="width: 0%;"></div>
                                </div>
                            </div>
                        `;
                        $container.append(itemHtml);
                    });
                    
                    // Animate the progress bars to their values
                    setTimeout(function () {
                        $container.find(".f1_progress_fill").each(function () {
                            const targetWidth = $(this).attr("data-width");
                            $(this).css("width", targetWidth);
                        });
                    }, 50);
                    
                    console.log(`[Metrics] Loaded F1 scores successfully for ${modelName}`);
                } else {
                    $container.html('<div style="text-align: center; font-size: 13px; color: rgba(255,255,255,0.4);">No metrics available</div>');
                }
            },
            error: function (xhr, status, error) {
                console.error("[Metrics] Failed to fetch model metrics:", error);
                $container.html('<div style="text-align: center; font-size: 13px; color: #ef4444;">Failed to load metrics</div>');
            }
        });
    }

    // Trigger metrics loading on dropdown selection change
    $(document).on("change", "#neuralModelSelect", function () {
        const modelName = $(this).val();
        loadModelMetrics(modelName);
    });

    // Load models automatically during dashboard setup
    loadAvailableModels();

    // -------------------------------------------------------------
    // 7. Dynamic Sample Video Selector API Loader
    // -------------------------------------------------------------
    function loadAvailableSamples() {
        $.ajax({
            url: API_BASE_URL + "/get-available-samples",
            type: "GET",
            dataType: "json",
            success: function (response) {
                if (response && response.samples && response.samples.length > 0) {
                    const $select = $("#sample_video_selection");
                    $select.empty();
                    response.samples.forEach(function (sample) {
                        $select.append(`<option value="${sample.filepath}">${sample.label}</option>`);
                    });
                    console.log(`[Samples] Successfully loaded ${response.samples.length} dynamic samples.`);
                }
            },
            error: function (xhr, status, error) {
                console.error("[Samples] Failed to fetch available samples:", error);
            }
        });
    }

    // Load samples automatically during dashboard setup
    loadAvailableSamples();

    // -------------------------------------------------------------
    // 8. Client-Side ONNX Runtime Web Local Inference Engine
    // -------------------------------------------------------------
    async function ensureOnnxRuntimeLoaded() {
        if (window.ort) return true;
        return new Promise((resolve, reject) => {
            console.log("[ONNX] Loading ONNX Runtime Web CDN dynamically...");
            const script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
            script.onload = () => {
                console.log("[ONNX] ONNX Runtime Web script loaded successfully!");
                resolve(true);
            };
            script.onerror = () => {
                reject(new Error("Failed to load ONNX Runtime Web from CDN."));
            };
            document.head.appendChild(script);
        });
    }

    let localClassNames = null;
    async function ensureClassNamesLoaded() {
        if (localClassNames) return localClassNames;
        return new Promise((resolve, reject) => {
            $.ajax({
                url: API_BASE_URL + "/download-ptl-model?filename=class_names.json",
                type: "GET",
                dataType: "json",
                success: function (data) {
                    localClassNames = data;
                    resolve(data);
                },
                error: function (err) {
                    reject(err);
                }
            });
        });
    }

    async function extractVideoFramesLocal(file, n_frames = 10, target_h = 112, target_w = 112) {
        return new Promise((resolve, reject) => {
            const video = document.createElement("video");
            video.preload = "auto";
            video.muted = true;
            video.playsInline = true;
            video.src = URL.createObjectURL(file);
            
            video.onloadedmetadata = async function() {
                const duration = video.duration;
                console.log(`[Decoder] Video metadata loaded. Duration: ${duration}s`);
                
                const frameTimestamps = [];
                // Sample uniformly across the video timeline
                for (let i = 0; i < n_frames; i++) {
                    frameTimestamps.push((duration / (n_frames + 1)) * (i + 1));
                }
                
                const canvas = document.createElement("canvas");
                canvas.width = target_w;
                canvas.height = target_h;
                const ctx = canvas.getContext("2d");
                
                const framesBuffer = new Float32Array(3 * n_frames * target_h * target_w);
                
                try {
                    for (let i = 0; i < n_frames; i++) {
                        const timestamp = frameTimestamps[i];
                        $("#progressPercent").text(Math.round(((i + 1) / n_frames) * 100) + "%");
                        $("#progress").css("width", Math.round(((i + 1) / n_frames) * 100) + "%");
                        
                        // Seek video to timestamp
                        await seekVideoToTimestamp(video, timestamp);
                        
                        // Draw frame with aspect-preserving zero-padding to target resolution
                        drawFrameWithAspectPad(video, ctx, target_h, target_w);
                        
                        // Extract frame pixels normalized to [0, 1]
                        const imgData = ctx.getImageData(0, 0, target_w, target_h);
                        const pixels = imgData.data; // RGBA
                        
                        // Pack into PyTorch Channel-First order: (C, T, H, W)
                        for (let y = 0; y < target_h; y++) {
                            for (let x = 0; x < target_w; x++) {
                                const pixelIdx = (y * target_w + x) * 4;
                                const r = pixels[pixelIdx] / 255.0;
                                const g = pixels[pixelIdx + 1] / 255.0;
                                const b = pixels[pixelIdx + 2] / 255.0;
                                
                                const r_offset = 0 * (n_frames * target_h * target_w) + i * (target_h * target_w) + y * target_w + x;
                                const g_offset = 1 * (n_frames * target_h * target_w) + i * (target_h * target_w) + y * target_w + x;
                                const b_offset = 2 * (n_frames * target_h * target_w) + i * (target_h * target_w) + y * target_w + x;
                                
                                framesBuffer[r_offset] = r;
                                framesBuffer[g_offset] = g;
                                framesBuffer[b_offset] = b;
                            }
                        }
                    }
                    
                    URL.revokeObjectURL(video.src);
                    resolve(framesBuffer);
                } catch (err) {
                    URL.revokeObjectURL(video.src);
                    reject(err);
                }
            };
            
            video.onerror = function() {
                reject(new Error("Failed to load video file for local decoding."));
            };
        });
    }

    function seekVideoToTimestamp(video, timestamp) {
        return new Promise((resolve) => {
            video.currentTime = timestamp;
            video.onseeked = function() {
                resolve();
            };
        });
    }

    function drawFrameWithAspectPad(video, ctx, target_h, target_w) {
        const video_w = video.videoWidth || 320;
        const video_h = video.videoHeight || 240;
        
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, target_w, target_h);
        
        const scale = Math.min(target_w / video_w, target_h / video_h);
        const new_w = Math.round(video_w * scale);
        const new_h = Math.round(video_h * scale);
        
        const x_offset = Math.round((target_w - new_w) / 2);
        const y_offset = Math.round((target_h - new_h) / 2);
        
        ctx.drawImage(video, x_offset, y_offset, new_w, new_h);
    }

    let onnxSession = null;
    async function runClientSideONNXInference(videoFile) {
        $("#select_container").addClass("hidden-container");
        $("#result_container").show();
        $("#gifImage").hide().attr("src", "");
        $("#spinnerWrapper").show();
        $("#processingText").show();
        $("#inferenceSuccessDisplay").hide();
        $("#progressBar").show();
        $("#progress").css("width", "0%");
        $("#progressPercent").text("0%");
        
        try {
            $("#processingText span").text("Initializing local WASM/WebGL execution environment...");
            await ensureOnnxRuntimeLoaded();
            
            $("#processingText span").text("Caching neural network action index mapping...");
            const classNames = await ensureClassNamesLoaded();
            
            $("#processingText span").text("Decoding and pre-processing spatiotemporal video frames...");
            const inputBuffer = await extractVideoFramesLocal(videoFile, 10, 112, 112);
            
            $("#processingText span").text("Initializing neural engine session (WebGL accelerated)...");
            if (!onnxSession) {
                const modelUrl = API_BASE_URL + "/download-ptl-model?filename=ucf101_paper_x112_b16_l2_d30_300k_best.onnx";
                onnxSession = await ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['webgl', 'wasm']
                });
            }
            
            $("#processingText span").text("Running local feed-forward model inference pass...");
            const inputTensor = new ort.Tensor('float32', inputBuffer, [1, 3, 10, 112, 112]);
            const results = await onnxSession.run({ video_input: inputTensor });
            
            const logits = results.prediction.data;
            const probs = softmax(logits);
            
            const maxIdx = probs.reduce((maxIdx, val, idx, arr) => val > arr[maxIdx] ? idx : maxIdx, 0);
            const confidence = probs[maxIdx];
            const predictedClass = classNames[maxIdx];
            const confidencePercent = Math.round(confidence * 100);
            
            const canvas = document.getElementById("videoThumbnailCanvas");
            const previewUrl = canvas.toDataURL("image/png");
            $("#gifImage").attr("src", previewUrl).fadeIn(400);
            
            $("#progressBar").hide();
            $("#spinnerWrapper").hide();
            $("#processingText").hide();
            
            $("#result").html(`Local Prediction: <span style="color: #10b981;">${predictedClass}</span>`);
            $("#confidenceValueText").text(confidencePercent + "%");
            $("#inferenceSuccessDisplay").fadeIn(400);
            
            setTimeout(function () {
                $("#confidenceBarFill").css("width", confidencePercent + "%");
            }, 200);
            
            // Dynamically load the metrics of this paper model inside f1ScoresContainer
            loadModelMetrics("ucf101_paper_x112_b16_l2_d30_300k_best.pth");
            
            showToast(
                "Local Inference Complete",
                `Identified: ${predictedClass} with ${confidencePercent}% confidence locally in-browser!`,
                "success"
            );
            
        } catch (err) {
            console.error("Local inference failed:", err);
            $("#progressBar").hide();
            $("#spinnerWrapper").hide();
            $("#processingText").hide();
            
            $("#gifImage").attr("src", "/static/img/work.gif").fadeIn();
            $("#result").html(`<span style="color:#ef4444;">Local Inference Failure</span>`);
            $("#confidenceValueText").text("0%");
            $("#confidenceBarFill").css("width", "0%");
            $("#inferenceSuccessDisplay").fadeIn(400);
            
            showToast("Local Inference Error", err.message || "Failed to execute ONNX session.", "error");
        }
    }

    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    // Notify user of active setup on reload
    console.log("[LUMAT] Client application initialized successfully.");
    showToast(
        "LUMAT Activated", 
        "Welcome to Lovely Unified Mindful Artificial Thought (LUMAT) dashboard.", 
        "success"
    );
});
