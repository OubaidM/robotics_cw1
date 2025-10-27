#!/usr/bin/env python3
"""
Live YOLO11 hot-desk detector – Gradio web UI
Usage:  python demo_app.py
Open:   http://localhost:7860
"""
import gradio as gr
from ultralytics import YOLO
import cv2, os, tempfile, time

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

def predict_image(img, conf, iou):
    """Return annotated PIL image + JSON summary"""
    # Convert numpy array to OpenCV format if necessary
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Resize the image
    img = resize_image(img)

    start = time.time()
    results = net.predict(source=img, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot()          # BGR numpy array
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    elapsed = time.time() - start
    boxes   = results[0].boxes 
    summary = {net.names[int(c)]: int((boxes.cls == c).sum())
               for c in boxes.cls.unique()}
    
    # Extract confidences and find the maximum
    confidences = boxes.conf.tolist()
    max_conf = max(confidences) if confidences else 0
    max_conf_perc = (max(confidences))*100 if confidences else 0
    confidence_text = f"Max Confidence: {max_conf_perc:.2f}%"

    return annotated, f"Inference: {elapsed*1000:.1f} ms\nObjects: {summary}\n{confidence_text}"

def predict_video(video_path, conf, iou, progress=gr.Progress()):
    """Process video frame-by-frame and return mp4"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.mktemp(suffix='.mp4')
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
