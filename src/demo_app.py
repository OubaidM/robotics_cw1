#!/usr/bin/env python3
"""
Live YOLO11 hot-desk detector – Gradio web UI
Usage:  python demo_app.py
Open:   http://localhost:7860
"""
import gradio as gr
from ultralytics import YOLO
import cv2, os, tempfile, time
from PIL import Image, ImageOps          # PIL handles EXIF automatically
import numpy as np
from pathlib import Path

MODEL_PATH = "model/hotdesk_final_model.pt"   # The best model path
model = MODEL_PATH if os.path.exists(MODEL_PATH) else "yolo11m.pt"

print("Loading model ...")
net = YOLO(model)

def resize_image(image, size=640):
    """Resize an image maintaining the aspect ratio, with the largest side being 'size'."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def auto_orient(im_pil: Image.Image) -> Image.Image:
    """Return upright PIL image; works for any EXIF orientation tag."""
    return ImageOps.exif_transpose(im_pil)

def predict_image(img, conf, iou):
    """Return annotated numpy RGB image + text summary."""
    # Normalize input into numpy RGB and also prepare a BGR copy for OpenCV/yolo
    if isinstance(img, Image.Image):
        pil = auto_orient(img)
        img_rgb = np.array(pil)
    elif isinstance(img, np.ndarray):
        img_rgb = img  # assume Gradio / caller provides RGB numpy
    elif isinstance(img, (str, Path)):
        pil = Image.open(img)
        pil = auto_orient(pil)
        img_rgb = np.array(pil)
    else:
        raise TypeError("predict_image() expects a PIL.Image, numpy.ndarray, or path-like input")

    # Handle alpha / grayscale -> ensure RGB
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:       # RGBA -> RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    elif img_rgb.ndim == 2:                              # Gray -> RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

    # Convert to BGR for OpenCV / model if needed
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Resize (keep API expecting BGR numpy if your resize_image uses cv2)
    img_bgr = resize_image(img_bgr)

    start = time.time()
    results = net.predict(source=img_bgr, imgsz=640, conf=conf, iou=iou, verbose=False, device=0) # Change device to "cpu" if having issues on GPU
    annotated_bgr = results[0].plot()          # BGR numpy array from model
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    elapsed = time.time() - start

    boxes = results[0].boxes
    summary = {net.names[int(c)]: int((boxes.cls == c).sum())
               for c in boxes.cls.unique()}

    confidences = boxes.conf.tolist() if hasattr(boxes, "conf") else []
    max_conf = max(confidences) if confidences else 0
    confidence_text = f"Max Confidence: {max_conf*100:.2f}%"

    return annotated_rgb, f"Inference: {elapsed*1000:.1f} ms\nObjects: {summary}\n{confidence_text}"

def predict_video(video_path, conf, iou, progress=gr.Progress()):
    """Process video frame-by-frame and return mp4"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.mkstemp(suffix='.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(frames):
        ret, frame = cap.read()
        if not ret: break
        results = net.predict(source=frame, conf=conf, iou=iou, verbose=False, device=0)
        annotated = results[0].plot()
        out.write(annotated)
        progress(idx/frames, desc="Processing video ...")

    cap.release(); out.release()
    return out_path

# ---------------- Gradio UI ----------------
with gr.Blocks(title="Hot-Desk Detector") as demo:
    gr.Markdown("# 🔍 Hot-Desk Object Detector (YOLO11)")
    gr.Markdown("Drag an image or video ⬇️, adjust confidence, then hit **Run**.")

    with gr.Row():
        conf = gr.Slider(0.05, 0.95, value=0.25, label="Confidence threshold")
        iou  = gr.Slider(0.10, 0.95, value=0.45, label="NMS IoU threshold")

    with gr.Tab("Image"):
        img_in  = gr.Image(type="numpy", label="Input")
        img_out = gr.Image(type="numpy", label="Output")
        json    = gr.Textbox(label="Stats", lines=2)
        img_btn = gr.Button("Run", variant="primary")
        img_btn.click(predict_image, inputs=[img_in, conf, iou], outputs=[img_out, json])

    with gr.Tab("Video"):
        vid_in  = gr.Video(label="Input video")
        vid_out = gr.Video(label="Output video")
        vid_btn = gr.Button("Run", variant="primary")
        vid_btn.click(predict_video, inputs=[vid_in, conf, iou], outputs=vid_out)

    gr.Examples(
        examples=[["sample_desk.jpg"]],
        inputs=img_in,
        outputs=[img_out, json],
        fn=predict_image,
        cache_examples=False,
    )

demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
