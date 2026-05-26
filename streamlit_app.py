import os, tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Detector", layout="wide", page_icon="!")
st.title("Object Detector")

st.sidebar.header("Configuration")
mode    = st.sidebar.radio("Mode", ["Image Upload", "Live Webcam"])
w_opts  = [f for f in ["yolov8x.pt", "yolov8m.pt", "yolov8n.pt"] if os.path.exists(f)]
weights = st.sidebar.selectbox("Model Weights", w_opts or ["yolov8x.pt"])
use_cnn = st.sidebar.toggle("CNN Feature Enhancement (ResNet50)", value=True)
conf    = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, 0.25, 0.05)
iou     = st.sidebar.slider("IoU Threshold", 0.10, 0.95, 0.45, 0.05)
imgsz   = st.sidebar.selectbox("Image Size", [320, 416, 512, 640], index=3)
import torch
st.sidebar.caption(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

@st.cache_resource(show_spinner="Loading model...")
def load_detector(w, cnn):
    from hybrid_detector import HybridObjectDetector
    return HybridObjectDetector(yolo_model=w, use_cnn_features=cnn)

def detection_table(results, names):
    boxes = results.boxes
    if not boxes or len(boxes) == 0:
        return None
    rows = []
    for box in boxes:
        cid = int(box.cls.item())
        rows.append({
            "Class": names[cid],
            "Confidence": round(float(box.conf.item()), 3),
            "x1": round(float(box.xyxy[0][0]), 1),
            "y1": round(float(box.xyxy[0][1]), 1),
            "x2": round(float(box.xyxy[0][2]), 1),
            "y2": round(float(box.xyxy[0][3]), 1),
        })
    return pd.DataFrame(rows)

if mode == "Image Upload":
    st.subheader("Image Upload Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(img, caption="Input Image", use_container_width=True)
        if st.button("Detect Objects"):
            detector = load_detector(weights, use_cnn)
            with st.spinner("Running hybrid detection..."):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img.save(tmp.name)
                    tmp_path = tmp.name
                results, cnn_feats, elapsed = detector.detect(tmp_path, conf=conf, iou=iou)
                os.unlink(tmp_path)
            annotated = Image.fromarray(results.plot()[..., ::-1])
            col2.image(annotated, caption="Detections", use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects Detected", len(results.boxes) if results.boxes else 0)
            m2.metric("Inference Time", f"{elapsed*1000:.1f} ms")
            m3.metric("FPS", f"{1/elapsed:.1f}")
            if use_cnn and cnn_feats is not None:
                st.info(f"CNN feature shape: {tuple(cnn_feats.shape)}")
            df = detection_table(results, detector.yolo.names)
            if df is not None:
                st.subheader(f"Detected {len(df)} object(s)")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df["Class"].value_counts())
            else:
                st.warning("No objects detected above the confidence threshold.")

elif mode == "Live Webcam":
    st.subheader("Live Webcam Detection")

    # TTA is too slow for real-time — warn user and suggest yolov8n for CPU
    if use_cnn:
        st.warning(
            "CNN rescoring adds latency. For smooth webcam FPS use `yolov8n.pt` "
            "or disable CNN Enhancement in the sidebar."
        )

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av, cv2
        from hybrid_detector import HybridObjectDetector

        RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class YOLOProcessor(VideoProcessorBase):
            def __init__(self):
                # disable TTA for webcam — 4x YOLO passes per frame kills FPS
                self.detector = HybridObjectDetector(
                    yolo_model=weights,
                    use_cnn_features=use_cnn,
                    use_tta=False,          # TTA off for real-time
                )
                self.conf      = conf
                self.iou       = iou
                self.det_count = 0
                self.fps       = 0.0
                self._last_annotated = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                try:
                    results, _, elapsed = self.detector.detect(
                        img, conf=self.conf, iou=self.iou
                    )
                    self.det_count = len(results.boxes) if results.boxes else 0
                    self.fps       = round(1 / max(elapsed, 1e-6), 1)
                    annotated      = results.plot()          # returns BGR numpy
                except Exception as e:
                    # on any error keep showing the raw frame so stream doesn't die
                    annotated = img.copy()
                    cv2.putText(annotated, f"Error: {e}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(annotated, "YOLOv8 + ResNet50", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"FPS: {self.fps}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Objects: {self.det_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        st.info("Click START to activate your webcam. Press STOP to end the stream.")
        st.caption("Tip: select `yolov8n.pt` in the sidebar for higher FPS on CPU.")

        ctx = webrtc_streamer(
            key="hybrid-webcam",
            video_processor_factory=YOLOProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            c1, c2 = st.columns(2)
            c1.metric("Objects (last frame)", ctx.video_processor.det_count)
            c2.metric("FPS",                  ctx.video_processor.fps)

    except ImportError:
        st.error("streamlit-webrtc is not installed.")
        st.code("pip install streamlit-webrtc av", language="bash")