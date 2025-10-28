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
- **mAP50 vs mAP50–95:** The difference between these two tells us about localization quality. The model’s mAP50 is 0.939, but mAP50–95 is 0.758 — this drop means boxes are close but not perfectly tight. It’s excellent at *finding* objects but slightly less precise in *positioning* them. Since the system will be used for classification, this level of performance is more than sufficient.  
- **IoU (Intersection over Union):** This measures how much the predicted box overlaps with the real one. A perfect box has IoU = 1.0. As IoU thresholds rise, only very tight boxes count as correct — which is why mAP50–95 is tougher.  
- **Fine-tuning hyperparameters for higher IoU performance:** This means adjusting how the model learns — such as lowering the learning rate, increasing the box loss weight, and reducing distortion in data augmentation. These tweaks help it produce tighter, more accurate bounding boxes.

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
4. **Dataset imbalance:** Pen class dominates (1970 samples) while bottle had only 360, but the model still generalized well across classes.  

---

## ⚗️ Extended Experimentation and Class Issues

While trying to improve the performance of the **bottle** class (which had limited samples), additional data and fine-tuning experiments were performed:

- **Added 120 new bottle images** to increase representation.  
- **Performed bottle-focused fine-tuning** using `copy_paste`, increased `box` and `dfl` loss weights.  
- **Attempted layer freezing** (`freeze=10`) to stabilize shared features while refining class boundaries.  
- **Tried partial unfreezing and reduced learning rates** (down to `1e-5`) for gentle head fine-tuning.

### 🧩 Outcome
These experiments did improve *bottle* and *pen* accuracy a lot, but also caused significant cross-class interference:
- False positives increased heavily for **headset**.  
- Confidence for **notebook** and other stable classes dropped notably.  
- Global confidence scores became inconsistent.  

As a result, the decision was made to **revert to the best stable checkpoint (`hotdesk_final_model.pt`)** for the final version.

---

## 🧪 Issue in Demo Application
In `demo_app.py`, detections involving **bottle** often produce low confidence or misclassification — this stems from the limited and unbalanced bottle data. Despite targeted fine-tuning, results remained inconsistent, and improvements to one class tended to harm others.

---

## 🚀 Future Work
To ensure long-term stability and reliability:
1. **Option 1 – Remove the bottle class entirely** from training and dataset configuration to maintain balanced detection across the remaining classes.  
2. **Option 2 – Rebuild from scratch with a more balanced dataset**, ensuring each class has roughly equal samples and consistent labeling.  
3. Explore **class-weighted training** or **synthetic augmentation** (CutMix, CopyPaste) for future versions, but only with sufficient data diversity.

---

**In short:**  
Attempts to improve the underrepresented *bottle* class demonstrated the trade-offs in fine-tuning object detectors. Enhancing one class without enough balanced data led to degraded overall performance. The final model prioritizes stability and reliability across all classes, as expected for a classification-focused deployment.
