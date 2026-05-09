function inspectionPage(config) {
    return {
        hasResult: Boolean(config.hasResult),
        resultId: config.resultId || "",
        blendApi: config.blendApi,
        localScoreApi: config.localScoreApi,
        defaultHeatmapUrl: config.defaultHeatmapUrl || "",
        defaultBlendUrl: config.defaultBlendUrl || "",
        measurementMode: config.measurementMode || "auto_ratio",
        patternMode: config.patternMode || "auto",
        cameraSupported: Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        cameraActive: false,
        cameraError: "",
        videoStream: null,
        videoDevices: [],
        selectedDeviceId: "",
        supportFiles: [],
        supportCaptureCount: 0,
        currentHeatmapSrc: config.defaultBlendUrl || config.defaultHeatmapUrl || "",
        blendEnabled: Boolean(config.hasResult),
        blendAlpha: Number(config.defaultAlpha || 0.45),
        modalOpen: false,
        probeData: {
            x: 0,
            y: 0,
            local_score: 0,
            suggested_defect: "",
        },

        init() {
            this.currentHeatmapSrc = this.blendEnabled
                ? (this.defaultBlendUrl || this.defaultHeatmapUrl)
                : this.defaultHeatmapUrl;
            this.initCamera();
        },

        async initCamera() {
            if (!this.cameraSupported) {
                return;
            }

            await this.refreshCameraDevices();
            const self = this;
            window.addEventListener("beforeunload", () => self.stopCamera());
        },

        async refreshCameraDevices() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                this.videoDevices = devices.filter((device) => device.kind === "videoinput");
                if (!this.selectedDeviceId && this.videoDevices.length > 0) {
                    this.selectedDeviceId = this.videoDevices[0].deviceId;
                }
            } catch (error) {
                this.cameraError = "Unable to list camera devices.";
            }
        },

        async startCamera() {
            if (!this.cameraSupported) {
                this.cameraError = "Camera access is not supported in this browser.";
                return;
            }

            this.cameraError = "";
            if (this.videoStream) {
                this.stopCamera();
            }

            const constraints = {
                video: this.selectedDeviceId
                    ? { deviceId: { exact: this.selectedDeviceId } }
                    : true,
                audio: false,
            };

            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                this.videoStream = stream;
                const video = document.getElementById("live-camera-video");
                if (video) {
                    video.srcObject = stream;
                    await video.play();
                }
                this.cameraActive = true;
                await this.refreshCameraDevices();
            } catch (error) {
                this.cameraError = error && error.message
                    ? error.message
                    : "Camera access denied.";
                this.cameraActive = false;
            }
        },

        stopCamera() {
            if (this.videoStream) {
                this.videoStream.getTracks().forEach((track) => track.stop());
            }
            this.videoStream = null;
            this.cameraActive = false;
        },

        changeCamera() {
            if (this.cameraActive) {
                this.startCamera();
            }
        },

        captureMainImage() {
            this.captureToInput("main", true);
        },

        captureSupportImage() {
            this.captureToInput("support", false);
        },

        captureToInput(target, autoSubmit) {
            if (!this.cameraActive) {
                window.alert("Start the camera before capturing.");
                return;
            }

            const video = document.getElementById("live-camera-video");
            if (!video || !video.videoWidth) {
                window.alert("Camera preview is not ready yet.");
                return;
            }

            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (!blob) {
                    window.alert("Capture failed. Please try again.");
                    return;
                }

                const file = new File(
                    [blob],
                    `capture_${Date.now()}.jpg`,
                    { type: "image/jpeg" }
                );

                if (target === "support") {
                    this.addSupportFile(file);
                } else {
                    this.setMainFile(file);
                    if (autoSubmit) {
                        this.submitInspection();
                    }
                }
            }, "image/jpeg", 0.92);
        },

        submitInspection() {
            const form = document.getElementById("inspection-form");
            if (form) {
                form.submit();
            }
        },

        setMainFile(file) {
            const input = document.getElementById("inspection-image-input");
            if (!input) {
                return;
            }

            const transfer = new DataTransfer();
            transfer.items.add(file);
            input.files = transfer.files;
        },

        addSupportFile(file) {
            const input = document.getElementById("support-images-input");
            if (!input) {
                return;
            }

            this.supportFiles = this.supportFiles.concat([file]);
            const transfer = new DataTransfer();
            this.supportFiles.forEach((item) => transfer.items.add(item));
            input.files = transfer.files;
            this.supportCaptureCount = this.supportFiles.length;
        },

        clearSupportFiles() {
            this.supportFiles = [];
            this.supportCaptureCount = 0;
            const input = document.getElementById("support-images-input");
            if (input) {
                input.value = "";
            }
        },

        syncSupportFiles(event) {
            const files = Array.from(event.target.files || []);
            this.supportFiles = files;
            this.supportCaptureCount = files.length;
        },

        async updateBlend() {
            if (!this.hasResult) {
                return;
            }

            if (!this.blendEnabled) {
                this.currentHeatmapSrc = this.defaultHeatmapUrl;
                return;
            }

            try {
                const response = await fetch(this.blendApi, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        result_id: this.resultId,
                        alpha: Number(this.blendAlpha),
                    }),
                });

                const payload = await response.json();
                if (!response.ok) {
                    throw new Error(payload.error || "Failed to generate blended heatmap.");
                }

                this.currentHeatmapSrc = `${payload.url}?t=${Date.now()}`;
            } catch (error) {
                window.alert(error.message || "Blend request failed.");
            }
        },

        async requestLocalScore(event) {
            if (!this.hasResult) {
                return;
            }

            const image = event.currentTarget;
            const rect = image.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            try {
                const response = await fetch(this.localScoreApi, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        result_id: this.resultId,
                        x,
                        y,
                        viewWidth: rect.width,
                        viewHeight: rect.height,
                    }),
                });

                const payload = await response.json();
                if (!response.ok) {
                    throw new Error(payload.error || "Failed to fetch local score.");
                }

                this.probeData = payload;
                this.modalOpen = true;
            } catch (error) {
                window.alert(error.message || "Heatmap probe failed.");
            }
        },
    };
}
