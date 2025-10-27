# Error Analysis — YOLO11m (Final Model, Explained Version)

This section explains the model’s results in plain language so that collaborators and readers can understand both the performance and what the metrics really mean.

---

## 📊 Summary of Key Metrics
| Metric | Meaning | Value |
|---------|----------|-------|
| **mAP50** | Accuracy when we count a detection as correct if its box overlaps ≥ 50% with the true box. Measures “did it find the object?” | 0.939 |
| **mAP50–95** | Average accuracy across many overlap thresholds (0.50 → 0.95). Measures how *precisely* boxes match true objects. | 0.758 |
| **Precision** | Out of all detected objects, how many were correct. | 0.935 |
| **Recall** | Out of all real objects, how many were found. | 0.909 |
| **F1 Score** | Harmonic mean of Precision & Recall — overall detection balance. | 0.922 |
| **Macro-F1** | Average of F1 per class (treating all classes equally). | ≈ 0.920 |

---

## 🧠 Understanding the Metrics
- **mAP50 vs mAP50–95:** The difference between these two tells us about localization quality. The model’s mAP50 is 0.939, but mAP50–95 is 0.758 — this drop means boxes are close but not perfectly tight. It’s excellent at *finding* objects but slightly less precise in *positioning* them but since it will be used for classification, it is perfectly alright.
- **IoU (Intersection over Union):** This measures how much the predicted box overlaps with the real one. A perfect box has IoU = 1.0. As IoU thresholds rise, only very tight boxes count as correct — which is why mAP50–95 is tougher.
- **Fine-tuning hyperparameters for higher IoU performance:** This means tweaking how the model learns — such as lowering the learning rate, increasing the box loss weight, and reducing distortion in data augmentation. These small adjustments help the model learn to place bounding boxes more accurately, improving performance at stricter IoU levels.

---

## 📈 Per-Class Performance (AP50)
| Class | AP50 | Comment |
|--------|-------|----------|
| mug | 0.809 | Some label noise and high variation caused lower accuracy |
| headset | 0.850 | Misses some small or occluded headsets |
| mouse | 0.944 | Consistent detections |
| stapler | 0.995 | Excellent; consistent object shape |
| notebook | 0.988 | Very strong and consistent |
| pen | 0.959 | Balanced and well-detected |
| phone | 0.988 | Excellent localization |
| bottle | 0.977 | Performs well despite few samples |

---

## 🔍 Observations
1. **Mug class:** Most errors come from inconsistent bounding boxes or visual diversity — e.g., shiny mugs, different angles, partial occlusions.
2. **Headset:** Sometimes missed when small or partially out of frame.
3. **Minor localization drift:** The 0.18 drop between mAP50 and mAP50–95 indicates small shifts in box placement. Boxes are slightly off-center or larger than needed.
4. **Dataset imbalance:** Pen class dominates (1970 samples) while bottle has only 360, but the model still generalizes well.

---

## 🛠️ How to Improve Further
1. Recheck mug labels and add more varied examples.
2. Reduce geometric augmentations (like strong mosaic or perspective) in the final fine-tune to stabilize localization.
3. Slightly increase the `box` loss weight (e.g., from 7.5 → 9.0) and lower the learning rate to focus on fine alignment.
4. Increase training resolution (e.g., 768 px) for the final tuning stage to capture finer details.

---

**In short:** the model finds objects very reliably, and only needs minor refinements to make its bounding boxes even more precise.
