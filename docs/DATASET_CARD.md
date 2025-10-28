# Dataset Card — Hot-Desk Office Items

**Link to dataset**: https://drive.google.com/file/d/1AMjAsiS05qCvisMhd-GK3zwGGi4cEE6D/view?usp=sharing

**Classes:** mug, headset, mouse, stapler, notebook, pen, phone, bottle  
**Total instances:** 8,345  
**Imbalance ratio:** 5.47× (pen most frequent, bottle least)

| Class | Instances | Notes |
|--------|------------|-------|
| mug | 1224 | Manually verified for labeling errors |
| headset | 1559 | Some small/angled variations |
| mouse | 1132 | Stable samples |
| stapler | 426 | Few examples, consistent |
| notebook | 780 | Consistent, easily separable |
| pen | 1970 | Dominant class |
| phone | 894 | Diverse samples |
| bottle | 360 | Least represented |

**Annotation format:** YOLO — class x_center y_center width height (normalized).

**Paths:**
```
train: /home/hotdesk/yolo_docker_project/images/train
val:   /home/hotdesk/yolo_docker_project/images/valid
test:  /home/hotdesk/yolo_docker_project/images/test
```

**Note:** 118 new instances was added to bottle class to try and balance/fine-tune the model. But this did not work as expected and had more drawbacks than benefit. The submitted model was trained on the above dataset.

